from net_modules.deconv import *
from net_modules.deconv import _deconv2d
from net_modules.deconv import _kernel
from net_modules.deconv import _stride
from net_modules.deconv import get2d_deconv_output_size
from net_modules.ndeconv import _deconv_mask
from prettytensor import pretty_tensor_class as prettytensor

import zutils.tf_math_funcs as tmf


@prettytensor.Register()
def nearest_upsampling(
        input_layer, kernel, stride, edges=PAD_SAME, name=PROVIDED
):

    assert len(input_layer.shape) == 4, "input rank must be 4"

    kernel = _kernel(kernel)
    stride = _stride(stride)

    input_height = input_layer.shape[1]
    input_width = input_layer.shape[2]
    depth = input_layer.shape[3]

    filter_height = kernel[0]
    filter_width = kernel[1]

    row_stride = stride[1]
    col_stride = stride[2]

    out_rows, out_cols = get2d_deconv_output_size(
        input_height, input_width, filter_height, filter_width, row_stride, col_stride, edges)

    output_shape_3d = [input_layer.shape[0], out_rows, out_cols, depth, 1]

    kernel_3d = kernel + [1]
    stride_3d = stride + [1]

    filter_mask = tf.ones(shape=kernel_3d + [1, 1], dtype=input_layer.dtype)
    output_tensor = tf.nn.conv3d_transpose(
        value=tf.expand_dims(input_layer, axis=-1),
        filter=filter_mask, output_shape=output_shape_3d,
        strides=stride_3d, padding=edges, name=name
    )
    output_tensor = tf.squeeze(output_tensor, axis=4)
    if filter_height != row_stride or filter_width != col_stride:
        output_mask = _deconv_mask(
            tmf.get_shape(input_layer), output_shape_3d[:-1],
            kernel, stride, edges, input_layer.dtype)
        output_tensor = tf.multiply(output_tensor, output_mask)

    return input_layer.with_tensor(output_tensor)



