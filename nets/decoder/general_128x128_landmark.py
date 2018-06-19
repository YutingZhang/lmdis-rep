import tensorflow as tf
import zutils.tf_math_funcs as tmf
import numpy as np
import zutils.pt_utils as ptu
import prettytensor as pt
from net_modules.common import *

from net_modules.hourglass import hourglass

from net_modules.auto_struct.keypoint_decoder import Factory as BaseFactory


class Factory(BaseFactory):

    @staticmethod
    def pt_defaults_scope_value():
        return {
            # non-linearity
            'activation_fn': default_activation.current_value,
            # for batch normalization
            'batch_normalize': True,
            'learned_moments_update_rate': 0.0003,
            # 'learned_moments_update_rate': 1.,
            'variance_epsilon': 0.001,
            'scale_after_normalization': True
        }

    default_patch_feature_dim = 8

    def __init__(self, recon_dist_param_num=1, options=None):
        super().__init__(recon_dist_param_num, options)
        if "image_channels" in options:
            self.image_channels = options["image_channels"]
        else:
            self.image_channels = 3

    def image_size(self):
        return 64, 64

    def feature2image(self, feature_map):
        output_channels = 3*self.recon_dist_param_num
        hgd = [
            {"type": "conv2d", "depth": 64},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 32x32
            {"type": "conv2d", "depth": 128},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 16x16
            {"type": "conv2d", "depth": 256},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 8x8
            {"type": "conv2d", "depth": 512},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 4x4
            {"type": "conv2d", "depth": 512},
        ]
        with pt.defaults_scope(**self.pt_defaults_scope_value()):
            output_tensor = hourglass(
                feature_map, hgd,
                net_type=self.options["hourglass_type"] if "hourglass_type" in self.options else None,
                extra_highlevel_feature=None
            )
            output_tensor = (
                pt.wrap(output_tensor).
                deconv2d(3, 32, stride=2).
                conv2d(3, output_channels, activation_fn=None)
            ).tensor
        return output_tensor

    def input_feature_dim(self):
        return 64

    def bg_feature2image(self, bg_feature):
        batch_size = tmf.get_shape(bg_feature)[0]
        with pt.defaults_scope(**self.pt_defaults_scope_value()):
            return (
                pt.wrap(bg_feature).
                conv2d(3, 512).                 # 2
                deconv2d(3, 512, stride=2).     # 4
                deconv2d(3, 256, stride=2).     # 8
                deconv2d(3, 256, stride=2).     # 16
                deconv2d(3, 128, stride=2).     # 32
                deconv2d(3, 64, stride=2).      # 64
                deconv2d(3, 32, stride=2).      # 128
                conv2d(3, 3*self.recon_dist_param_num, activation_fn=None).tensor
            )



