from zutils.py_utils import value_class_for_with
import zutils.tf_math_funcs as tmf


default_activation = value_class_for_with(tmf.leaky_relu)

import net_modules.nearest_upsampling
_prettytensor2 = net_modules.nearest_upsampling.prettytensor

import net_modules.deconv
_prettytensor3 = net_modules.deconv.prettytensor

import net_modules.ndeconv
_prettytensor4 = net_modules.ndeconv.prettytensor

import net_modules.pixel_bias
_prettytensor5 = net_modules.pixel_bias.prettytensor

import net_modules.spatial_transformer_pt
_prettytensor6 = net_modules.spatial_transformer_pt.prettytensor

import net_modules.pt_patch_batch_normalization
_prettytensor7 = net_modules.pt_patch_batch_normalization.prettytensor

import net_modules.pt_group_connected
_prettytensor9 = net_modules.pt_group_connected.prettytensor
