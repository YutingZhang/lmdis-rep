from prettytensor.bookkeeper import *
from prettytensor.bookkeeper import _bare_var_name
from zutils.pt_utils import default_phase, pt
from prettytensor import pretty_tensor_class as prettytensor


# this is a monkey patch for exponential_moving_average in order to improve batch normalization


class PatchedBookkeeper(Bookkeeper):

    old_exponential_moving_average = Bookkeeper.exponential_moving_average

    def exponential_moving_average(
            self,
            var,
            avg_var=None,
            decay=0.999,
            ignore_nan=False
    ):
        """Calculates the exponential moving average.
        TODO(): check if this implementation of moving average can now
        be replaced by tensorflows implementation.
        Adds a variable to keep track of the exponential moving average and adds an
        update operation to the bookkeeper. The name of the variable is
        '%s_average' % name prefixed with the current variable scope.
        Args:
           var: The variable for which a moving average should be computed.
           avg_var: The variable to set the average into, if None create a zero
             initialized one.
           decay: How much history to use in the moving average.
             Higher, means more history values [0, 1) accepted.
           ignore_nan: If the value is NaN or Inf, skip it.
        Returns:
           The averaged variable.
        Raises:
          ValueError: if decay is not in [0, 1).
        """

        with self._g.as_default():
            if decay < 0 or decay >= 1.0:
                raise ValueError('Decay is %5.2f, but has to be in [0, 1).' % decay)
            if avg_var is None:
                avg_name = '%s_average' % _bare_var_name(var)
                with tf.control_dependencies(None):
                    with tf.name_scope(avg_name + '/Initializer/'):
                        if isinstance(var, tf.Variable):
                            init_val = var.initialized_value()
                        elif var.get_shape().is_fully_defined():
                            init_val = tf.constant(0,
                                                   shape=var.get_shape(),
                                                   dtype=var.dtype.base_dtype)
                        else:
                            init_val = tf.constant(0, dtype=var.dtype.base_dtype)
                    avg_var = tf.Variable(init_val, name=avg_name, trainable=False)

            avg_name = _bare_var_name(avg_var)
            num_updates = tf.get_variable(
                name='%s_numupdates' % avg_name, shape=(),
                dtype=tf.int64, initializer=tf.constant_initializer(0, dtype=tf.int64),
                trainable=False
            )
            is_running = tf.get_variable(
                name='%s_isrunning' % avg_name, shape=(),
                dtype=tf.bool, initializer=tf.constant_initializer(True, dtype=tf.bool),
                trainable=False
            )

            exact_avg_mode = tf.group(
                tf.assign(is_running, False), tf.assign(num_updates, 0), tf.variables_initializer([avg_var])
            )
            running_avg_mode = tf.assign(is_running, True)

            # op to switch between running and non-running mode
            avg_var.is_switchable_avg = True
            avg_var.exact_avg_mode = exact_avg_mode
            avg_var.running_avg_mode = running_avg_mode
            avg_var.num_updates = num_updates

            # compute decay

            num_updates, is_running, _ = tf.tuple([num_updates, is_running, tf.assign_add(num_updates, 1)])

            # -------------- the following control does not work------ Looks like a tensorflow bug when using Variables
            # with tf.control_dependencies([tf.assign_add(num_updates, 1)]):
            #     num_updates = tf.identity(num_updates)
            # --------------------------------------------------------

            num_updates_f = tf.cast(num_updates, tf.float32)
            running_decay = tf.minimum(
                decay,
                tf.maximum(0.9, (1.0 + num_updates_f) / (10.0 + num_updates_f))
            )
            exact_decay = (num_updates_f - 1.)/num_updates_f
            decay = tf.where(is_running, running_decay, exact_decay)
            # decay = tf.check_numerics(decay, "avg_decay_failed", "%s_avg_decay" % avg_name)

            # apply average
            with tf.device(avg_var.device):
                if ignore_nan:
                    var = tf.where(tf.is_finite(var), var, avg_var)
                if var.get_shape().is_fully_defined():
                    avg_update = tf.assign_sub(avg_var, (1 - decay) * (avg_var - var))
                else:
                    avg_update = tf.assign(avg_var,
                                           avg_var - (1 - decay) * (avg_var - var),
                                           validate_shape=False)
            self._g.add_to_collection(GraphKeys.UPDATE_OPS, avg_update)

            return avg_update

Bookkeeper.exponential_moving_average = PatchedBookkeeper.exponential_moving_average
