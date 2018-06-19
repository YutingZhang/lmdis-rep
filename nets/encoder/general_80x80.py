import numpy as np
import tensorflow as tf
import prettytensor as pt
from net_modules.common import *
from net_modules import keypoints_2d
import math

from net_modules.hourglass import hourglass

import zutils.tf_math_funcs as tmf
from net_modules.auto_struct.keypoint_encoder import Factory as BaseFactory

def encoder_map(input_tensor, hourglass_type=None):

    hgd = [
        {"type": "conv2d", "depth": 32, "decoder_depth": 64},
        {"type": "conv2d", "depth": 64, "decoder_depth": 64},
        {"type": "skip", "layer_num": 2},
        {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 40 x 40
        {"type": "conv2d", "depth": 128},
        {"type": "skip", "layer_num": 2},
        {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 20x20
        {"type": "conv2d", "depth": 256},
        {"type": "skip", "layer_num": 2},
        {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 10x10
        {"type": "conv2d", "depth": 512},
    ]

    output_tensor = hourglass(
        input_tensor, hgd,
        net_type=hourglass_type
    )
    return output_tensor




class Factory(BaseFactory):
    @staticmethod
    def pt_defaults_scope_value():
        return {
            # non-linearity
            'activation_fn': default_activation.current_value,
            # for batch normalization
            'batch_normalize': True,
            'learned_moments_update_rate': 0.0003,
            'variance_epsilon': 0.001,
            'scale_after_normalization': True
        }

    default_patch_feature_dim = 8

    def __init__(self, output_channels, options):
        """
        :param output_channels: output_channels for the encoding net
        """
        super().__init__(output_channels, options)

        self.target_input_size = [96, 96]
        self.input_size = [80, 80]

    def image2heatmap(self, image_tensor):
        hgd = [
            {"type": "conv2d", "depth": 32, "decoder_depth": self.options["keypoint_num"] + 1,
             "decoder_activation_fn": None},
            # plus one for bg
            {"type": "conv2d", "depth": 32},
            {"type": "skip", "layer_num": 3, },
            {"type": "pool", "pool": "max"},
            {"type": "conv2d", "depth": 64},
            {"type": "conv2d", "depth": 64},
            {"type": "skip", "layer_num": 3, },
            {"type": "pool", "pool": "max"},
            {"type": "conv2d", "depth": 64},
            {"type": "conv2d", "depth": 64},
            {"type": "skip", "layer_num": 3, },
            {"type": "pool", "pool": "max"},
            {"type": "conv2d", "depth": 64},
            {"type": "conv2d", "depth": 64},
        ]

        with pt.defaults_scope(**self.pt_defaults_scope_value()):
            raw_heatmap = hourglass(
                image_tensor, hgd,
                net_type=self.options["hourglass_type"] if "hourglass_type" in self.options else None
            )
            # raw_heatmap = pt.wrap(raw_heatmap).pixel_bias(activation_fn=None).tensor

        return raw_heatmap



    def image2feature(self, image_tensor):

        if self.patch_feature_dim == 0:
            return None
        
        hgd = [
            {"type": "conv2d", "depth": 32, "decoder_depth": 64},
            {"type": "conv2d", "depth": 64, "decoder_depth": 64},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 40 x 40
            {"type": "conv2d", "depth": 128},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 20x20
            {"type": "conv2d", "depth": 256},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 10x10
            {"type": "conv2d", "depth": 512},
        ]

        with pt.defaults_scope(**self.pt_defaults_scope_value()):
            feature_map = hourglass(
                image_tensor, hgd,
                net_type=self.options["hourglass_type"] if "hourglass_type" in self.options else None
            )
        return feature_map

