import prettytensor as pt
import tensorflow as tf
import numpy as np

from copy import copy

import zutils.tf_math_funcs as tmf

from zutils.option_struct import OptionDef

from easydict import EasyDict as edict


class HourglassOptions:
    @staticmethod
    def layer(p):
        p["type"] = "conv2d"    # "max_pool", "average_pool"
        if p["type"] == "conv2d":
            p["kernel"] = 3
            p["depth"] = None
            assert p["depth"] is not None, "depth must be set for convolutional layer"
            p["decoder_depth"] = None
            p["stride"] = 1
            p["encoder"] = True
            p["decoder"] = True
            p["padding"] = "SAME"
            p["activation_fn"] = "default"      # "default" for prettytensor default
            p["decoder_activation_fn"] = "default"      # "default" for prettytensor default
        elif p["type"] == "pool":
            p["pool"] = "max"  # "max", "average"
            p["kernel"] = 2
            p["stride"] = p["kernel"]
            p["padding"] = "VALID"
        elif p["type"] == "dropout":
            p["keep_prob"] = 0.5
            p["encoder"] = True
            p["decoder"] = False
        elif p["type"] == "skip":
            p["kernel"] = 3
            p["layer_num"] = 3
            p["keep_prob"] = 1
        else:
            raise ValueError("Hourglass options: Unknown type")

        p.finalize()


def _hourglass_builder(x, layer_def, net_type=None, extra_highlevel_feature=None):

    if net_type is None:
        net_type = "hourglass"
    assert net_type in ("hourglass", "multiscale_encoder", "multiscale_decoder")

    # complete the options
    encoder_layer_stack = list()
    for the_layer_def in layer_def:
        the_opt = OptionDef(the_layer_def, HourglassOptions)
        the_layer_opt = the_opt["layer"]
        opt = the_layer_opt.get_edict()
        encoder_layer_stack.append(opt)

    # build encoder
    input_size = tmf.get_shape(x)[1:3]
    output_depth = None
    output_activation_fn = None
    current_decoder_depth = tmf.get_shape(x)[-1]
    current_pointer = pt.wrap(x)
    decoder_layer_stack = list()
    for l in encoder_layer_stack:
        d = copy(l)
        if l.type == "conv2d":
            if not type == "multiscale_decoder" and l.encoder:
                extra_conv_args = dict()
                if l.activation_fn != "default":
                    extra_conv_args["activation_fn"] = l.activation_fn
                current_pointer = current_pointer.conv2d(
                    l.kernel, l.depth, stride=l.stride, edges=l.padding, **extra_conv_args
                )
            if l.decoder:
                if l.decoder_depth is None:
                    d.decoder_depth = current_decoder_depth
                delattr(d, "depth")
                delattr(d, "activation_fn")
                current_decoder_depth = l.depth
                d = [d]
                if output_depth is None:
                    output_depth = l.decoder_depth
                    output_activation_fn = l.decoder_activation_fn
            else:
                d = []

            if l.stride != 1:
                if l.padding == "SAME":
                    upsampling_parameter = {
                        "type": "pool", "kernel": l.stride, "stride": l.stride, "padding": l.padding
                    }
                else:
                    upsampling_parameter = {
                        "type": "pool", "kernel": l.kernel, "stride": l.stride, "padding": l.padding
                    }
                upsampling_parameter = edict(upsampling_parameter)
                d.append(upsampling_parameter)

        elif l.type == "pool":

            if l.pool == "max":
                current_pointer = current_pointer.max_pool(
                    l.kernel, stride=l.stride, edges=l.padding
                )
            elif l.pool in ("average", "ave"):
                current_pointer = current_pointer.average_pool(
                    l.kernel, stride=l.stride, edges=l.padding
                )
            elif l.pool == "resize":
                assert l.kernel == l.stride and l.edges == "VALID", "inconsistent options with resize"
                if isinstance(l.kernel, (list, tuple)):
                    resize_kernel = l.kernel
                else:
                    resize_kernel = [l.kernel, l.kernel]
                current_pointer = pt.wrap(tf.image.resize_images(
                    current_pointer, [
                        round(tmf.get_shape(current_pointer)[1] / resize_kernel[0]),
                        round(tmf.get_shape(current_pointer)[2] / resize_kernel[1])
                    ], method=tf.image.ResizeMethod.BILINEAR))
            else:
                raise ValueError("unrecognized pool type")

        elif l.type == "dropout":
            if not type == "multiscale_decoder" and l.encoder:
                current_pointer = current_pointer.dropout(l.keep_prob)
            if not l.decoder:
                d = None

        elif l.type == "skip":
            d.input_tensor = current_pointer.tensor

        else:
            raise ValueError("Unknown layer type for hourglass")

        if d is not None:
            if isinstance(d, list):
                decoder_layer_stack.extend(d)
            else:
                decoder_layer_stack.append(d)

    if net_type == "multiscale_encoder":
        extra_conv_args = dict()
        if output_activation_fn != "default":
            extra_conv_args["activation_fn"] = output_activation_fn
        current_pointer = current_pointer.conv2d(1, output_depth, stride=1, edges="SAME", **extra_conv_args)
        current_pointer = pt.wrap(tf.image.resize_images(current_pointer, input_size))

    highlevel_shape = tmf.get_shape(current_pointer)
    highlevel_dim = np.prod(highlevel_shape[1:])
    if extra_highlevel_feature is not None:
        current_pointer = pt.wrap(tf.add(
            current_pointer,
            pt.wrap(extra_highlevel_feature).fully_connected(highlevel_dim).reshape(highlevel_shape).tensor
        ))

    # build decoder
    current_decoder_depth = tmf.get_shape(current_pointer.tensor)[-1]
    decoder_layer_stack.reverse()
    for l in decoder_layer_stack:
        if l.type == "skip":
            cur_skip_pointer = pt.wrap(l.input_tensor)
            skip_input_depth = tmf.get_shape(l.input_tensor)[-1]
            skip_chain_depth = max(skip_input_depth, current_decoder_depth)
            for i in range(l.layer_num):
                if i < l.layer_num-1:
                    the_skip_chain_depth = skip_chain_depth
                else:
                    the_skip_chain_depth = current_decoder_depth
                cur_skip_pointer = cur_skip_pointer.conv2d(
                    l.kernel, the_skip_chain_depth, stride=1
                )
                if l.keep_prob < 1.:
                    cur_skip_pointer = cur_skip_pointer.dropout(l.keep_prob)
            if net_type == "multiscale_encoder":
                if tmf.get_shape(cur_skip_pointer)[1:3] != input_size:
                    cur_skip_pointer = pt.wrap(tf.image.resize_images(cur_skip_pointer, input_size))
            current_pointer = current_pointer.join(
                [cur_skip_pointer], join_function=tf.add_n
            )

        elif not net_type == "multiscale_encoder":

            if l.type == "conv2d":
                extra_deconv_args = dict()
                if l.decoder_activation_fn != "default":
                    extra_deconv_args["activation_fn"] = l.decoder_activation_fn
                current_pointer = current_pointer.deconv2d(
                    l.kernel, l.decoder_depth, stride=l.stride, edges=l.padding, **extra_deconv_args
                )
                current_decoder_depth = l.decoder_depth

            elif l.type == "dropout":
                current_pointer = current_pointer.dropout(l.keep_prob)

            elif l.type == "pool":
                if l.pool == "resize":
                    assert l.kernel == l.stride and l.edges == "VALID", "inconsistent options with resize"
                    if isinstance(l.kernel, (list, tuple)):
                        resize_kernel = l.kernel
                    else:
                        resize_kernel = [l.kernel, l.kernel]
                    current_pointer = pt.wrap(tf.image.resize_images(
                        current_pointer, [
                            round(tmf.get_shape(current_pointer)[1] * resize_kernel[0]),
                            round(tmf.get_shape(current_pointer)[2] * resize_kernel[1])
                        ], method=tf.image.ResizeMethod.BILINEAR))
                else:
                    current_pointer = current_pointer.nearest_upsampling(
                        l.kernel, stride=l.stride, edges=l.padding
                    )

            else:
                raise ValueError("Internal error: unrecognized layer type")

    return current_pointer.tensor


def hourglass_builder(x, layer_def, net_type=None, scope=None, extra_highlevel_feature=None):
    """
    """

    if scope is not None:
        with tf.variable_scope(scope):
            y = _hourglass_builder(x, layer_def, net_type=net_type, extra_highlevel_feature=extra_highlevel_feature)
    else:
        y = _hourglass_builder(x, layer_def, net_type=net_type, extra_highlevel_feature=extra_highlevel_feature)

    return y


hourglass = hourglass_builder
