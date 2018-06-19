from net_modules.deconv import *
from net_modules.spatial_transformer import transformer
from net_modules.tps_stn import TPS_STN, TPS_TRANSFORM
import numpy as np

import zutils.tf_math_funcs as tmf


@prettytensor.Register()
def spatial_transformer(
        input_layer, theta, out_size, name=PROVIDED
):

    # init
    input_shape = tmf.get_shape(input_layer.tensor)
    assert len(input_shape) == 4, "input tensor must be rank 4"
    if theta is np.ndarray:
        theta = tf.constant(theta)
    elif not tmf.is_tf_data(theta):
        theta = theta.tensor

    # apply transformer
    output = transformer(input_layer.tensor, theta, out_size=out_size, name=name)

    # make output shape explicit
    output = tf.reshape(output, [input_shape[0]]+out_size+[input_shape[3]])
    return output


@prettytensor.Register()
def coordinate_inv_transformer(
        input_layer, theta, name=PROVIDED
):

    # init
    input_tensor = input_layer.tensor
    input_shape = tmf.get_shape(input_tensor)
    assert len(input_shape) == 3, "input tensor must be rank 3"
    if theta is np.ndarray:
        theta = tf.constant(theta)
    elif not tmf.is_tf_data(theta):
        theta = theta.tensor

    keypoint_num = tmf.get_shape(input_tensor)[1]

    with tf.variable_scope(name):
        kp2_e = tf.concat([input_tensor, tf.ones_like(input_tensor[:, :, :1])], axis=2)
        kp2_e = tf.expand_dims(kp2_e, axis=-1)
        transform_e = tf.tile(tf.expand_dims(theta, axis=1), [1, keypoint_num, 1, 1])
        kp1from2_e = tf.matmul(transform_e, kp2_e)
        kp1from2 = tf.squeeze(kp1from2_e, axis=-1)

    return kp1from2


@prettytensor.Register()
def spatial_transformer_tps(
        input_layer, nx, ny, cp, out_size, fp_more=None, name=PROVIDED
):

    # init
    input_shape = tmf.get_shape(input_layer.tensor)
    assert len(input_shape) == 4, "input tensor must be rank 4"

    def convert_to_tensor(a):
        if a is np.ndarray:
            a = tf.constant(a)
        elif not tmf.is_tf_data(a):
            a = a.tensor
        return a

    cp = convert_to_tensor(cp)

    batch_size = tmf.get_shape(input_layer)[0]

    # apply transformer
    with tf.variable_scope(name):
        cp = tf.reshape(cp, [batch_size, -1, 2])
        output = TPS_STN(input_layer.tensor, nx, ny, cp, out_size=out_size, fp_more=fp_more)

    # make output shape explicit
    output = tf.reshape(output, [input_shape[0]]+out_size+[input_shape[3]])
    return output


@prettytensor.Register()
def coordinate_inv_transformer_tps(
        input_layer, nx, ny, cp, fp_more=None, name=PROVIDED
):
    input_shape = tmf.get_shape(input_layer.tensor)
    assert len(input_shape) == 3, "input tensor must be rank 3"
    p = input_layer.tensor
    with tf.variable_scope(name):
        output = TPS_TRANSFORM(nx, ny, cp, p, fp_more=fp_more)
    return output
