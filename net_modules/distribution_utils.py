import tensorflow as tf
import zutils.tf_math_funcs as tmf


# handle extended structure_param, pathc_feature_x, overall_features
def reshape_extended_features(a, param_factor):
    if a is None:
        return None, None
    kept_shape = tmf.get_shape(a)[:-1] + [tmf.get_shape(a)[-1] // param_factor]
    a = tf.reshape(a, kept_shape + [param_factor])
    main_chooser = (slice(None, None),) * len(kept_shape) + (0,)
    b = a[main_chooser]
    return a, b
