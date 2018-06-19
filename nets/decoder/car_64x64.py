import prettytensor as pt
import tensorflow as tf
from net_modules.common import *

from net_modules.hourglass import hourglass

import zutils.tf_math_funcs as tmf

import zutils.pt_utils as ptu

from net_modules.auto_struct.generic_decoder import Factory as GenericDecoderFactory
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

    def input_feature_dim(self):
        return 64

    def feature2image(self, feature_tensor):
        output_channels = 3*self.recon_dist_param_num

        hgd = [
            {"type": "conv2d", "depth": 64, "decoder_depth": output_channels, "decoder_activation_fn": None},
            {"type": "conv2d", "depth": 64, "decoder_depth": 32},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 40 x 40
            {"type": "conv2d", "depth": 128, "decoder_depth": 64},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 20x20
            {"type": "conv2d", "depth": 256},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 10x10
            {"type": "conv2d", "depth": 512},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 5x5
            {"type": "conv2d", "depth": 512},
        ]


        with pt.defaults_scope(**self.pt_defaults_scope_value()):
            output_tensor = hourglass(
                feature_tensor, hgd,
                net_type=self.options["hourglass_type"] if "hourglass_type" in self.options else None,
                extra_highlevel_feature=None
                )
        return output_tensor

    rotate_dominating_features_if_necessary = GenericDecoderFactory.rotate_dominating_features_if_necessary



