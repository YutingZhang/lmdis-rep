import tensorflow as tf
import datetime
import numpy as np
import zutils.tf_math_funcs as tmf

from zutils.py_utils import *
from scipy.io import savemat

class OneEpochRunner:

    def __init__(
            self, data_module, output_list=None,
            net_func=None, batch_axis=0, num_samples=None, disp_time_interval=2,
            output_fn=None, is_large=False):

        self.data_module = data_module
        self.num_samples = self.data_module.num_samples()
        self.batch_axis = batch_axis
        self.disp_time_interval = disp_time_interval
        self.output_fn = output_fn
        self.is_large = is_large

        if num_samples is not None:
            if self.num_samples < num_samples:
                print("specified number_samples is larger than one epoch")
            else:
                self.num_samples = num_samples

        self.use_net_func = output_list is None  # otherwise use net_func
        if self.use_net_func:
            assert net_func is not None, \
                "output_list and net_func should not be both specified"
            self.net_func = net_func
            # remark: net_func(sess)
        else:
            assert net_func is None, \
                "one of output_list and net_func must be specified"
            self.output_list = output_list
            [self.flatten_output_list, self.output_wrap_func] = \
                recursive_flatten_with_wrap_func(
                    lambda x: tmf.is_tf_data(x), self.output_list)

        self.data_module.reset()
        self.cur_sample_end = 0

    def run_single_batch(self, sess):

        if self.cur_sample_end >= self.num_samples:
            return None

        if self.use_net_func:
            output_val = self.net_func(sess)
        else:
            output_val = sess.run(self.flatten_output_list, {})
            output_val = self.output_wrap_func(output_val)

        batch_size = first_element_apply(
            lambda x: isinstance(x, np.ndarray),
            lambda x: x.shape[self.batch_axis], output_val)
        self.batch_size = batch_size

        new_end = self.cur_sample_end + batch_size
        if new_end > self.num_samples:
            effective_batch_size = \
                batch_size - (new_end-self.num_samples)
            slice_indexes = (slice(None),)*self.batch_axis + (slice(effective_batch_size),)
            output_val = recursive_apply(
                lambda x: isinstance(x, np.ndarray),
                lambda x: x[slice_indexes], output_val)
        self.cur_sample_end = new_end
        return output_val

    def run(self, sess):
        disp_countdown = IfTimeout(self.disp_time_interval)
        num_samples_total = self.num_samples

        output_val_single = self.run_single_batch(sess)
        output_val = []

        while output_val_single is not None:
            output_val += [output_val_single]

            iter = self.data_module.iter()
            if self.data_module.epoch() == 0:
                num_samples_finished = self.data_module.num_samples_finished()
            else:
                num_samples_finished = self.num_samples

            if disp_countdown.is_timeout():
                epoch_percentage = num_samples_finished / num_samples_total * 100
                print("%s] Iter %d (%4.1f%% = %d / %d)" %
                      (datetime.datetime.now().strftime('%Y-%m/%d-%H:%M:%S.%f'),
                       iter, epoch_percentage, num_samples_finished, num_samples_total))
                disp_countdown = IfTimeout(self.disp_time_interval)

            
            if self.is_large and (num_samples_finished % (100*self.batch_size) == 0 or num_samples_finished == self.num_samples):
                output_val = recursive_apply(
                    lambda *args: isinstance(args[0], np.ndarray),
                    lambda *args: np.concatenate(args, axis=self.batch_axis),
                    *output_val)
                self.dir_path = os.path.dirname(self.output_fn+'_'+'%06d'%num_samples_finished)
                if not os.path.exists(self.dir_path):
                    os.makedirs(self.dir_path)
                savemat(self.output_fn+'_'+'%06d'%num_samples_finished+'.mat',output_val)
                print('Saving part of output to '+ self.output_fn+'_'+'%06d'%num_samples_finished+'.mat')
                output_val = []
            output_val_single = self.run_single_batch(sess)
 
        if not self.is_large:
            output_val = recursive_apply(
                lambda *args: isinstance(args[0], np.ndarray),
                lambda *args: np.concatenate(args, axis=self.batch_axis),
                *output_val)
            savemat(self.output_fn + ".mat", output_val)
            print('Saving output to ' + self.output_fn + ".mat")


