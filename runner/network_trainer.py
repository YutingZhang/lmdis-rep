import tensorflow as tf
import prettytensor as pt

import numpy as np

from zutils.py_utils import *

import zutils.tf_graph_utils as tgu
import zutils.tf_math_funcs as tmf


class NeuralNetworkTrainer:

    def __init__(
            self, data_module, loss_tensor, solver_type,
            solver_kwargs=None, disp_tensor_dict=None,
            minimizer_kwargs=None, update_ops=None,
            max_epochs=None, disp_time_interval=2, disp_prefix=None,
            learning_rate=None, global_step=None,
            snapshot_func=None, snapshot_interval=7200, snapshot_sharing=None,
            permanent_snapshot_step_list=None, snapshot_step_list=None,
            test_func=None, test_steps=10000, logger=None, scope=None,
            extra_output_tensors=None
    ):

        if solver_kwargs is None:
            solver_kwargs = dict()

        if minimizer_kwargs is None:
            minimizer_kwargs = dict()

        if disp_tensor_dict is None:
            disp_tensor_dict = dict()
        minimizer_kwargs = copy(minimizer_kwargs)

        if learning_rate is not None:
            solver_kwargs["learning_rate"] = learning_rate
        else:
            assert hasattr(solver_kwargs, "learning_rate"), "learning rate is not set"
        self.learning_rate_tensor = solver_kwargs["learning_rate"]
        if not tmf.is_tf_data(self.learning_rate_tensor):
            self.learning_rate_tensor = tf.constant(self.learning_rate_tensor)
        self.learning_rate = None

        if scope is None:
            scope = "trainer"

        self.data_module = data_module
        optimizer_func = getattr(tf.train, solver_type + "Optimizer")

        self.optimizer = optimizer_func(**solver_kwargs)

        # figure out subiters
        if "var_list" in minimizer_kwargs:
            var_list = minimizer_kwargs["var_list"]
        else:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self.update_shared = tgu.update_shared_vars(var_list)

        var_list = list(set(var_list) - set(tgu.get_freeze_collection()))   # remove freeze variables

        self.var_list = var_list
        minimizer_kwargs["var_list"] = var_list

        # cache variables

        old_variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # define training iters
        with tf.device("/cpu:0"), tf.variable_scope(scope):
            self.iter_variable = tf.Variable(
                0, trainable=False, dtype=tf.int64, name="trainer_step")
            self.pos_variable = tf.Variable(
                0, trainable=False, dtype=tf.int64, name="trainer_pos")

        # function for handling update ops
        def attach_updates_to_train_op(train_op_without_updates):
            # add update ops (mainly for batch normalization)
            if update_ops is None:
                train_op = pt.with_update_ops(train_op_without_updates)
            else:
                assert isinstance(update_ops, list), "update_ops must be a list"
                if update_ops:
                    train_op = tf.group(train_op_without_updates, *update_ops)
                else:
                    train_op = train_op_without_updates
            return train_op

        # define minimizer
        self.gradient_tensors = OrderedDict()

        is_single_device = tmf.is_tf_data(loss_tensor)
        assert is_single_device, \
            "ERROR: this code does not support multiple devices. Use CUDA_VISIBLE_DEVICES=... to specify the GPU."

        raw_gradient_tensor = self.optimizer.compute_gradients(loss_tensor, **minimizer_kwargs)

        # disp and extra variables
        self.loss_tensor = loss_tensor
        self.disp_tensor_dict = flatten_str_dict(disp_tensor_dict)
        self.extra_output_tensors = extra_output_tensors

        new_variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # train for all subiters
        self.gradient_tensor = []
        for g, v in raw_gradient_tensor:
            if hasattr(v, "lr_mult"):
                g *= v.lr_mult
            self.gradient_tensor.append((g,v))
        self.train_op_without_updates = self.optimizer.apply_gradients(self.gradient_tensor)
        self.train_op = attach_updates_to_train_op(self.train_op_without_updates)
        with tf.control_dependencies([self.update_shared]):
            self.train_op = tf.group(self.train_op)

        # sanity check
        assert "extra" not in disp_tensor_dict, \
            "extra is reserved for extra outputs"

        # saveable variables
        self.saveable_variables = list(set(new_variable_list) - set(old_variable_list))

        # helper for extra outputs
        self.extra_output_tensors_flattened, self.extra_output_tensors_wrapfunc = \
            recursive_flatten_with_wrap_func(tmf.is_tf_data, self.extra_output_tensors)

        # setup loss summaries
        self.loss_summaries = []
        with tf.name_scope('trainer_summaries'):
            if disp_prefix is not None:
                summary_prefix = disp_prefix + "_"
            else:
                summary_prefix = ""
            self.loss_summaries.append(
                tf.summary.scalar(summary_prefix+"Loss", self.loss_tensor))
            for k, v in self.disp_tensor_dict.items():
                self.loss_summaries.append(tf.summary.scalar(summary_prefix + k, v))
            self.loss_summaries.append(tf.summary.scalar(summary_prefix + "learning_rate", self.learning_rate_tensor))
        self.merged_loss_summaries = tf.summary.merge([self.loss_summaries]) 
        self.logger = logger  # do not do anything with it just set it up if possible

        # self.train_op = pt.apply_optimizer(
        #     self.optimizer, losses=[loss_tensor], **minimizer_kwargs)

        # step up variables for the training stage
        self.max_epochs = max_epochs
        self.disp_time_interval = disp_time_interval

        self.disp_prefix = disp_prefix
        self.total_iter = np.uint64(0)
        self.total_pos = np.uint64(0)

        if global_step is None:
            global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step

        with tf.device("/cpu:0"):
            self.iter_variable_inc = tf.assign_add(self.iter_variable, 1)
            self.global_step_inc = tf.assign_add(self.global_step, 1)
            self.pos_assign_placeholder = tf.placeholder(tf.int64, shape=[], name="trainer_pos_assign")
            self.pos_variable_assign = tf.assign(self.pos_variable, self.pos_assign_placeholder)

        # set up variables for run_init
        self.all_output_tensors = None
        self.all_output_names = None
        self.tmp_training_losses = None
        self.disp_countdown = None
        self.outside_timestamp = None
        self.tmp_iter_start = None

        # set up snapshot saver
        if permanent_snapshot_step_list is None:
            permanent_snapshot_step_list = []
        if snapshot_step_list is None:
            snapshot_step_list = []

        if snapshot_sharing is None:
            self.snapshot_runner_shared = False
            if snapshot_func is None:
                snapshot_interval = None
            else:
                if snapshot_interval is not None:
                    print("  - snapshot in every %d sec" % snapshot_interval)

            self.permanent_snapshot_condition = \
                ArgsSepFunc(lambda the_step: the_step in permanent_snapshot_step_list)
            self.snapshot_condition = \
                ArgsSepFunc(lambda the_step: the_step in snapshot_step_list)

            _snapshot_periodic_runner = PeriodicRun(
                snapshot_interval, snapshot_func
            )
            _snapshot_periodic_runner.add_extra_true_condition(
                self.snapshot_condition
            )
            _snapshot_periodic_runner.add_extra_true_condition(
                self.need_stop
            )
            _snapshot_periodic_runner.add_extra_true_condition(
                self.permanent_snapshot_condition,
                extra_func=lambda sess, step: snapshot_func(sess, step, preserve=True)
            )
            self.snapshot_periodic_runner = _snapshot_periodic_runner
        else:
            self.snapshot_runner_shared = True
            self.snapshot_periodic_runner = snapshot_sharing.snapshot_periodic_runner
            self.permanent_snapshot_condition = snapshot_sharing.permanent_snapshot_condition
            self.snapshot_condition = snapshot_sharing.snapshot_condition

        # set up test func
        self.test_func = test_func
        self.test_steps = test_steps

        # step up variables for avg update
        self.avg_var_list = None
        self.avg_var_forward_steps = None
        self.avg_var_minimum_update_steps = None
        self.avg_var_update_num = None
        self.avg_var_exact_mode = None
        self.avg_var_running_mode = None

    def set_logger(self, logger):
        self.logger = logger

    def run_init(self):
        self.all_output_tensors = \
            [self.loss_tensor] + list(self.disp_tensor_dict.values()) + self.extra_output_tensors_flattened
        self.all_output_names = ["Loss"] + list(self.disp_tensor_dict.keys())

        self.tmp_training_losses = np.asarray([0.0] * len(self.all_output_names), dtype=np.float64)
        # remark: higher precision for being more accurate
        self.disp_countdown = IfTimeout(self.disp_time_interval)
        self.tmp_iter_start = 0

        if not self.snapshot_runner_shared:
            self.snapshot_periodic_runner.reset()

    def need_stop(self):
        return self.data_module is not None and self.data_module.epoch() >= self.max_epochs

    def step(self, sess):

        if self.all_output_tensors is None:
            self.run_init()
            tf.train.start_queue_runners(sess=sess)

        if self.need_stop():
            return None

        if self.outside_timestamp is not None:
            outside_time = time.time() - self.outside_timestamp
            self.disp_countdown.add_ignored_time(time_amount=outside_time)

        if self.total_iter == 0:
            # resume data module stats (currently only support "resumable_wrapper")
            self.total_iter, self.total_pos = sess.run([self.iter_variable, self.pos_variable])
            if self.data_module is not None:
                self.data_module.set_pos(self.total_pos)

        run_outputs = sess.run(
            [self.train_op, self.merged_loss_summaries] + self.all_output_tensors +
            [self.learning_rate_tensor, self.iter_variable_inc, self.global_step_inc], {})
        global_step = run_outputs[-1]
        self.total_iter = run_outputs[-2]
        self.learning_rate = run_outputs[-3]
        loss_summaries = run_outputs[1]
        main_outputs = run_outputs[2:-3]
        losses = main_outputs[:len(self.all_output_names)]
        extra_outputs = self.extra_output_tensors_wrapfunc(main_outputs[len(self.all_output_names):])
        losses = np.asarray(losses, dtype=np.float64)     # remove the output from train object
        self.tmp_training_losses += losses
        output_dict = OrderedDict(zip(self.all_output_names, losses))
        output_dict["extra"] = extra_outputs

        if self.logger is not None:
            self.logger.add_summary(loss_summaries, global_step)

        if self.data_module is not None:
            self.total_pos = sess.run(self.pos_variable_assign, {self.pos_assign_placeholder: self.data_module.pos()})

        if self.disp_countdown.is_timeout():

            actual_time_interval = self.disp_countdown.interval

            iter_interval = self.total_iter - self.tmp_iter_start
            self.tmp_training_losses /= iter_interval

            iter_per_sec = iter_interval / actual_time_interval

            results = OrderedDict(zip(self.all_output_names, self.tmp_training_losses.tolist()))
            results_str = " \t".join(["%s: %.5g" % (k, v) for k, v in results.items()])
            timestamp_str = time_stamp_str()
            if self.disp_prefix is not None:
                timestamp_str = timestamp_str + "[" + self.disp_prefix + "]"
            if self.data_module is None:
                print("%s[lr:%g] step %d, iter %d (%g iter/sec):\t%s" % (
                    timestamp_str,
                    self.learning_rate,
                    global_step,
                    self.total_iter,
                    iter_per_sec,
                    results_str
                ))
            else:
                epoch_percentage = self.data_module.num_samples_finished() / \
                                   self.data_module.num_samples() * 100
                # results_str = list(chain(*results_str))  # equivalent to [j for i in result_str for j in i]
                # results_str = "".join(results_str)
                print("%s[lr:%g] step %d, iter %d (%4.1f%%, epoch %d, %g iter/sec):\t%s" %
                      (timestamp_str,
                       self.learning_rate,
                       global_step,
                       self.total_iter, epoch_percentage,
                       self.data_module.epoch(),
                       iter_per_sec,
                       results_str))

            self.tmp_training_losses[:] = 0.0
            self.tmp_iter_start = self.total_iter
            self.disp_countdown = IfTimeout(self.disp_time_interval)

        self.permanent_snapshot_condition.set_args(global_step)
        self.snapshot_condition.set_args(global_step)
        is_snapshotted, _ = self.snapshot_periodic_runner.run_if_timeout_with_prefixfunc(
            lambda: self.update_avgvar_if_necessary(sess=sess), sess, global_step
        )

        self.outside_timestamp = time.time()

        if self.test_func is not None:
            if is_snapshotted or (self.test_steps is not None and global_step % self.test_steps == 0):
                if not is_snapshotted:
                    self.update_avgvar_if_necessary(sess)
                _ = self.test_func(sess, global_step)

        if self.need_stop():
            self.update_avgvar(sess=sess)

        return output_dict

    def forward_step(self, sess):
        return sess.run(self.loss_tensor)

    def run(self, sess):

        assert self.max_epochs is not None, \
            "cannot run with a particular max_epochs"

        while self.step(sess) is not None:
            pass

    # for batch normalization

    def setup_avgvar_update(
            self, full_var_list=None, forward_steps=0, minimum_update_steps=0
    ):
        self.avg_var_list = None
        self.avg_var_forward_steps = None
        self.avg_var_minimum_update_steps = None
        if full_var_list is None or not full_var_list:
            return
        if forward_steps <= 0:
            return
        the_var_list = []
        for v in full_var_list:
            if hasattr(v, "is_switchable_avg") and v.is_switchable_avg:
                the_var_list.append(v)

        if not the_var_list:
            return

        the_var_list = list(set(the_var_list))

        the_update_nums = []
        the_exact_avg_modes = []
        the_running_avg_modes = []
        for v in the_var_list:
            the_update_nums.append(tf.reshape(v.num_updates, [1]))
            the_exact_avg_modes.append(v.exact_avg_mode)
            the_running_avg_modes.append(v.running_avg_mode)

        self.avg_var_list = the_var_list
        self.avg_var_forward_steps = forward_steps
        self.avg_var_minimum_update_steps = minimum_update_steps
        self.avg_var_update_num = tf.reduce_max(tf.concat(the_update_nums, axis=0))
        self.avg_var_exact_mode = tf.group(*the_exact_avg_modes)
        self.avg_var_running_mode = tf.group(*the_running_avg_modes)

    def update_avgvar_if_necessary(self, sess):
        if self.avg_var_list is None:
            return
        if (
            self.avg_var_minimum_update_steps <= 0 or
            sess.run(self.avg_var_update_num) > self.avg_var_minimum_update_steps
        ):
            self.update_avgvar(sess)

    def update_avgvar(self, sess, forward_steps=None):
        if self.avg_var_list is None:
            return
        if forward_steps is None:
            forward_steps = self.avg_var_forward_steps
        assert forward_steps is not None, "must set forward_steps"

        print("%s START UPDATE AVERAGE VARIABLES ======" % time_stamp_str())
        sess.run([self.avg_var_exact_mode])
        the_disp_countdown = IfTimeout(self.disp_time_interval)
        for k in range(forward_steps):
            if the_disp_countdown.is_timeout():
                print("%s update average variables: %d / %d" % (time_stamp_str(), k+1, forward_steps))
                the_disp_countdown = IfTimeout(self.disp_time_interval)
            self.forward_step(sess)
        sess.run([self.avg_var_running_mode])
        print("%s END UPDATE AVERAGE VARIABLES ======" % time_stamp_str())


# handle multiple devices (remark: only support a single device in this code) -------------------------------------

class TrainingNetOutput:
    def __init__(self):

        self.loss = None
        self.display_outputs = None
        self.device_outputs = OrderedDict()
        self.device_outputs["ps"] = None
        self.ps_device_outputs = None


def _single_device_training_net(data_tensors, train_net_func, default_reuse=None):
    # single default device cases
    print(" - training net on default device")
    loss, display_outputs, ps_device_outputs = train_net_func(data_tensors, default_reuse)
    output_entry = TrainingNetOutput()

    output_entry.loss = loss
    output_entry.display_outputs = display_outputs
    output_entry.device_outputs["ps"] = ps_device_outputs
    output_entry.ps_device_outputs = ps_device_outputs

    new_variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    return output_entry, new_variable_list


def single_device_training_net(data_tensors, train_net_func):
    """ generate training nets for multiple devices
    
    :param data_tensors: [ [batch_size, ...], [batch_size, ...], ... ]
    :param train_net_func: loss, display_outputs, first_device_output = train_net_func(data_tensors, default_reuse)
                Remark: loss can be a list or a dict, then the return of this function can be organized accordingly
    :return output_entry: a class instance with fields for a single device
    :return unique_variable_list: variable_list on the first/only device
    """

    old_variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    output_struct, new_variable_list = _single_device_training_net(data_tensors, train_net_func)

    unique_variable_list = list(set(new_variable_list) - set(old_variable_list))

    return output_struct, unique_variable_list

