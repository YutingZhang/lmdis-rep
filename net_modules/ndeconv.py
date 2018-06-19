from net_modules.deconv import *
from net_modules.deconv import _deconv2d
from net_modules.deconv import _kernel
from net_modules.deconv import _stride

import net_modules.ndeconv


@prettytensor.Register(
    assign_defaults=('activation_fn', 'l2loss', 'stddev', 'batch_normalize'))
class ndeconv2d:
    def __init__(self):
        self._deconv_internal = None

    def __call__(
        self,
        input_layer,
        kernel,
        depth,
        name=PROVIDED,
        stride=None,
        activation_fn=None,
        l2loss=None,
        init=None,
        stddev=None,
        bias=True,
        edges=PAD_SAME,
        batch_normalize=False
    ):

        input_shape = input_layer.shape
        self._deconv_internal = _deconv2d()
        output_layer = self._deconv_internal(
            input_layer,
            kernel, depth, name, stride, activation_fn, l2loss,
            init, stddev, bias, edges, batch_normalize)
        output_shape = output_layer.shape

        output_mask = _deconv_mask(input_shape, output_shape, kernel, stride, edges, output_layer.dtype)
        output_layer.with_tensor(tf.multiply(output_layer.tensor, output_mask))

        return output_layer


def _deconv_mask(input_shape, output_shape, kernel, stride, padding, dtype):
    with tf.variable_scope("fida_factor"):
        with tf.device("/cpu:0"):
            filter_mask = tf.ones(shape=_kernel(kernel) + [1, 1], dtype=dtype)
            input_mask = tf.ones(shape=[1] + input_shape[1:3] + [1], dtype=dtype)
        output_mask = tf.nn.conv2d_transpose(
            value=input_mask, filter=filter_mask, output_shape=[1] + output_shape[1:3] + [1],
            strides=_stride(stride), padding=padding
        )
    output_mask = tf.reduce_mean(output_mask) / output_mask
    return output_mask
