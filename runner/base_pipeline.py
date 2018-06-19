from abc import ABCMeta, abstractmethod
import tensorflow as tf
import os
import datetime
import pickle
import shutil
import glob
import stat

import zutils.tf_graph_utils as tgu

from zutils.option_struct import OptionStruct

from zutils.py_utils import *

ValueScheduler = tgu.ValueScheduler


class Pipeline:

    ValueScheduler = tgu.ValueScheduler

    __metaclass__ = ABCMeta

    # Abstract interfaces ---------------------------------------------------------

    @abstractmethod
    def resume(self, sess):
        pass

    @abstractmethod
    def test(self, sess, output_dir, is_snapshot=False):
        pass

    @property
    def opt_dict_raw(self):
        return self._opt_dict_raw

    @property
    def opt_dict(self):
        return self._opt_dict

    @property
    def opt(self):
        return self._opt

    # initialization ----------------------------------------------------------------------
    def __init__(
            self, model_dir=None, auto_save_hyperparameters=True, use_logging=True
    ):

        # save hyper parameters
        self.auto_save_hyperparameters = auto_save_hyperparameters
        self.use_logging = use_logging

        # model dir
        self.model_dir = model_dir
        self.log_dir = None
        self.snapshot_dir = None
        self.test_dir = None
        if self.model_dir is not None:
            if self.use_logging:
                self.log_dir = os.path.join(self.model_dir, "logs")
                self.snapshot_dir = os.path.join(self.model_dir, "model")
                self.test_dir = os.path.join(self.model_dir, "test.snapshot")

        # snapshot function
        if self.snapshot_dir is not None:
            self.periodic_snapshot_func = lambda sess, step, preserve=False: self.snapshot(
                sess, os.path.join(self.snapshot_dir, "snapshot"), step, preserve=preserve)
        else:
            self.periodic_snapshot_func = None

        # set the graph
        self.graph = tf.Graph()

        # global step
        with self.graph.as_default(), tf.device("/cpu:0"):
            self.global_step = tf.train.create_global_step()

        # hold necessary field
        self.net_saver = None
        self.saver = None
        self.logger = None

        # internal variables
        self._opt = None
        self._opt_dict = None
        self._opt_dict_raw = None

    def set_options(self, options):
        if isinstance(options, OptionStruct):
            self._opt_dict_raw = options.get_dict()
            option_name = options.option_name
        elif isinstance(options, dict):
            self._opt_dict_raw = options
            option_name = None
        else:
            raise ValueError("Unrecongized class for options")

        with self.graph.as_default():
            self._opt_dict = recursive_apply(
                lambda x: isinstance(x, tgu.ValueScheduler),
                lambda x: tgu.value_scheduler_to_tensor(x),
                self._opt_dict_raw
            )
        self._opt = dict2namedtuple(self._opt_dict, tuple_name=option_name)

        if self.auto_save_hyperparameters:
            self.save_hyperparameters()

    def init_logger(self):
        with self.graph.as_default():
            # training logs
            if self.log_dir is not None:
                print("* Define logger")
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.logger = tf.summary.FileWriter(self.log_dir, graph=self.graph)
            else:
                self.logger = None

    @staticmethod
    def create_session():
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        return sess

    # variables and states
    def initialize_variables(self, sess):
        # initializer
        trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_name)
        init = tf.variables_initializer(trainable_variables)
        sess.run(init)

    # basic snapshot and restore ---------------------------------------------------------------------

    def restore(
            self, sess, checkpoint_dir=None, checkpoint_filename=None,
            net_only=False, no_error=False, frozen_only=False
    ):
        if checkpoint_dir is None:
            checkpoint_dir = self.snapshot_dir
            assert checkpoint_dir is not None, "must specify checkpoint_dir"

        if checkpoint_filename is not None:
            if not isinstance(checkpoint_filename, str):
                checkpoint_filename = 'snapshot_step_%d' % checkpoint_filename

        def get_checkpoint_path(cdir, cfilename):
            if checkpoint_filename is not None:
                cpath = os.path.join(cdir, cfilename)
                if os.path.isfile(cpath + ".index"):
                    return cpath
            else:
                ckpt = tf.train.get_checkpoint_state(cdir)
                if ckpt and ckpt.model_checkpoint_path:
                    return ckpt.model_checkpoint_path
            return None

        model_checkpoint_path = get_checkpoint_path(checkpoint_dir, checkpoint_filename)
        if model_checkpoint_path is not None:
            # Restores from checkpoint
            if frozen_only:
                self.net_saver.load_weights(
                    sess, model_checkpoint_path, ignored_scope_level=1, var_list=tgu.get_freeze_collection()
                )
                print("restored FROZEN ONLY from %s" % model_checkpoint_path)
            elif net_only:
                self.net_saver.restore(sess, model_checkpoint_path)
                print("restored NET ONLY from %s" % model_checkpoint_path)
            else:
                self.saver.restore(sess, model_checkpoint_path)
                print("restored model from %s" % model_checkpoint_path)
        else:
            model_checkpoint_path = get_checkpoint_path(checkpoint_dir, checkpoint_filename)
            if frozen_only:
                self.net_saver.load_weights(
                    sess, model_checkpoint_path, ignored_scope_level=1, var_list=tgu.get_freeze_collection()
                )
                print("restored FROZEN ONLY from %s" % model_checkpoint_path)
            if model_checkpoint_path is not None:
                self.net_saver.restore(sess, model_checkpoint_path)
                print("restored NET ONLY from %s" % model_checkpoint_path)
            else:
                if checkpoint_filename is None:
                    err_msg = "no checkpoint file found in %s" % checkpoint_dir
                else:
                    err_msg = "no checkpoint file found as %s" % os.path.join(checkpoint_dir, checkpoint_filename)
                if not no_error:
                    raise ValueError(err_msg)
                print(err_msg)
                return False

        return True

    @staticmethod
    def _snapshot(sess, snapshot_path, saver, preserve=False):
        parent_path = os.path.dirname(snapshot_path)
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        saver.save(sess, snapshot_path)
        if preserve:
            snapshot_names = glob.glob(snapshot_path + "*")
            for fn in snapshot_names:
                s = os.stat(fn)[stat.ST_MODE]
                os.chmod(fn, s & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))  # make it read only

    def snapshot(self, sess, snapshot_path=None, step=None, preserve=False):
        if snapshot_path is None:
            snapshot_path = self.snapshot_dir
        assert snapshot_path is not None, "must specify snapshot_path"

        if step is not None:
            snapshot_path += "_step_%d" % step

        timestamp_str = datetime.datetime.now().strftime('%Y-%m/%d-%H:%M:%S.%f') + "]"
        print("%s snapshot to %s" % (timestamp_str, snapshot_path))

        self._snapshot(sess, snapshot_path, self.saver, preserve=preserve)
        netonly_path = \
            os.path.join(os.path.dirname(snapshot_path) + ".net_only", os.path.basename(snapshot_path))
        self._snapshot(sess, netonly_path, self.net_saver, preserve=preserve)

        timestamp_str = datetime.datetime.now().strftime('%Y-%m/%d-%H:%M:%S.%f') + "]"
        print("%s snapshot is done" % timestamp_str)

    def _snapshot_test(self, sess, global_step):
        output_dir = os.path.join(self.test_dir, "step_%d" % global_step)
        timestamp_str = datetime.datetime.now().strftime('%Y-%m/%d-%H:%M:%S.%f') + "]"
        print("%s run test: %s" % (timestamp_str, output_dir))

        assert hasattr(self, "run_full_test"), "no test function is provided"

        test_func = getattr(self, "run_full_test")
        test_output = test_func(sess, output_dir, is_snapshot=True)

        timestamp_str = datetime.datetime.now().strftime('%Y-%m/%d-%H:%M:%S.%f') + "]"
        print("%s test is done" % timestamp_str)

        return test_output

    # Common interfaces -----------------------------------------------------------

    def train_from_scratch(self, sess, finetune_list=None, load_frozen_only=False):
        self.initialize_variables(sess)
        self.load_per_finetune_list(sess, finetune_list, frozen_only=load_frozen_only)
        self.resume(sess)

    def load_per_finetune_list(self, sess, finetune_list, frozen_only=False):
        if finetune_list is None:
            return
        if isinstance(finetune_list, str):
            finetune_list = [finetune_list]
        for finetune_fn in finetune_list:
            ft_str = finetune_fn
            if frozen_only:
                var_list = tgu.get_freeze_collection()
                ft_str = "(FROZEN ONLY) " + ft_str
            else:
                var_list = None
            print("* finetune from %s" % ft_str)
            self.net_saver.load_weights(
                sess, finetune_fn, ignored_scope_level=1, var_list=var_list
            )
        self.net_saver.sync_to_worker_devices(sess)

    def save_hyperparameters(self, output_dir=None):
        if output_dir is None:
            output_dir = self.model_dir
        assert output_dir is not None, "must specify the output_dir"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if self.opt_dict_raw is not None:
            print('========== Save Hyper-parameters')
            param_fn = os.path.join(output_dir, "PARAM.p")
            if os.path.exists(param_fn):
                if os.path.exists(param_fn + ".bak"):
                    os.remove(param_fn + ".bak")
                shutil.move(param_fn, param_fn + ".bak")
            pickle.dump(self.opt_dict_raw, open(param_fn, "wb"))
            print('-- Done')
        else:
            print('Warning: tried to save the hyper-parameters, but could not find self.opt_dict')

    # full experiments (Training)
    def run_full_train(
            self, sess, output_dir=None, restore=False, restore_frozen_only=False,
            finetune_list=None, load_frozen_only=False
    ):

        if output_dir is None:
            output_dir = self.model_dir
        assert output_dir is not None, "must specify the output_dir"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('========== Train')
        from_scratch = True
        if restore:
            self.initialize_variables(sess)
            from_scratch = not self.restore(sess, no_error=True, frozen_only=restore_frozen_only)
        if from_scratch:
            self.train_from_scratch(sess, finetune_list=finetune_list, load_frozen_only=load_frozen_only)
        else:
            self.resume(sess)
        print('-- Done')

        print('========== Save Model')
        self.snapshot(sess, os.path.join(self.snapshot_dir, "model"))
        print('-- Done')

    def run_full_train_from_checkpoint(self, sess, output_dir=None, checkpoint_dir=None,checkpoint_filename=None, finetune_list=None, restore_frozen_only=False):

        if output_dir is None:
            output_dir = self.model_dir
        assert output_dir is not None, "must specify the output_dir"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('========== Train')
        self.restore(sess, checkpoint_dir=checkpoint_dir, net_only=False, checkpoint_filename=checkpoint_filename, frozen_only = restore_frozen_only)
        self.resume(sess)
        print('-- Done')

        print('========== Save Model')
        self.snapshot(sess, os.path.join(self.snapshot_dir, "model"))
        print('-- Done')

    def run_full_test(self, sess, output_dir=None, is_snapshot=False, is_large=False, save_img_data=True):

        tf.train.start_queue_runners(sess=sess)

        if output_dir is None:
            output_dir = os.path.join(self.model_dir, "test.final")
        assert output_dir is not None, "must specify the output_dir"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.test(sess, output_dir, is_snapshot, is_large, save_img_data)


    def run_full_test_from_checkpoint(
            self, sess, output_dir=None, checkpoint_dir=None, test_path=None, 
            snapshot_iter=None, is_large=False, save_img_data=True
    ):
        if output_dir is None:
            output_dir = self.model_dir
        assert output_dir is not None, "must specify the output_dir"

        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(output_dir, "model")
        else:
            assert output_dir is None, \
                "should not specify both output_dir and checkpoint_dir"

        self.restore(sess, checkpoint_dir, net_only=False, checkpoint_filename=snapshot_iter)

        if test_path is None:
            test_path = "test.final"

        if test_path[0] != "/":
            test_path = os.path.join(output_dir, test_path)

        self.run_full_test(sess, test_path, is_large=is_large, save_img_data=save_img_data)
        print("Test output: %s" % test_path)
 
