from net_modules.ndeconv import *
from net_modules.ndeconv import _deconv_mask
from prettytensor import parameters
from prettytensor.pretty_tensor_image_methods import conv2d as pt_conv2d

import zutils.tf_math_funcs as tmf
from net_modules.deconv import _kernel
from net_modules.deconv import _stride


class _conv2d(prettytensor.VarStoreMethod):
    __call__ = pt_conv2d.__call__


@prettytensor.Register(
    assign_defaults=('activation_fn', 'l2loss', 'batch_normalize',
                     'parameter_modifier', 'phase'))
class nconv2d:

    tmp_graph = tf.Graph()

    def __init__(self):
        self._internal_conv2d = _conv2d()

    def __call__(
        self,
        input_layer,
        kernel,
        depth,
        activation_fn=None,
        stride=(1, 1),
        l2loss=None,
        weights=None,
        bias=tf.zeros_initializer(),
        edges=PAD_SAME,
        batch_normalize=False,
        phase=prettytensor.Phase.train,
        parameter_modifier=parameters.identity,
        name=PROVIDED
    ):

        # compute output size
        input_shape = input_layer.shape
        input_mask_shape = [1]+input_shape[1:3]+[1]
        with self.tmp_graph.as_default(), tf.device("/cpu:0"):
            fake_output_mask = tf.nn.conv2d(
                input=tf.zeros(shape=input_mask_shape, dtype=tf.float32),
                filter=tf.zeros(shape=_kernel(kernel)+[1, 1], dtype=tf.float32),
                strides=_stride(stride),
                padding=edges
            )
            output_mask_shape = tmf.get_shape(fake_output_mask)

        # generate input mask
        input_mask = _deconv_mask(
            output_mask_shape, input_mask_shape, kernel, stride, edges, input_layer.dtype
        )

        input_layer.with_tensor(
            _gradient_scale(input_layer.tensor, input_mask)
        )

        output_layer = self._internal_conv2d(
            input_layer,
            kernel,
            depth,
            activation_fn,
            stride,
            l2loss,
            weights,
            bias,
            edges,
            batch_normalize,
            phase,
            parameter_modifier,
            name
        )

        return output_layer


def _gradient_scale(data, mask):

    data_grad_mask = (data - tf.stop_gradient(data) + 1.) * mask
    unscaled_data = tf.stop_gradient(data / mask) * data_grad_mask
    return unscaled_data
