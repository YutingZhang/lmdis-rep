import tensorflow as tf
import zutils.tf_math_funcs as tmf
import numpy as np
import zutils.pt_utils as ptu
import prettytensor as pt
from net_modules.common import *

from net_modules.hourglass import hourglass

from net_modules.auto_struct.keypoint_encoder import Factory as BaseFactory


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

        self.target_input_size = [192, 192]
        self.input_size = [128, 128]

    def image2heatmap(self, image_tensor):
        mid_tensor = (
            pt.wrap(image_tensor).
            conv2d(3, 32).
            max_pool(2, 2)
        ).tensor 

        hgd = [
            {"type": "conv2d", "depth": 64},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 32 x 32
            {"type": "conv2d", "depth": 128},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 16 x 16
            {"type": "conv2d", "depth": 256},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 8 x 8
            {"type": "conv2d", "depth": 512},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 4 x 4
            {"type": "conv2d", "depth": 512},
        ]

        with pt.defaults_scope(**self.pt_defaults_scope_value()):
            raw_heatmap_feat = hourglass(
                mid_tensor, hgd,
                net_type = self.options["hourglass_type"] if "hourglass_type" in self.options else None
            )

        return raw_heatmap_feat

    def image2feature(self, image_tensor):

        if self.patch_feature_dim == 0:
            return None

        mid_tensor = (
            pt.wrap(image_tensor).
            conv2d(3, 32).
            max_pool(2, 2)
        ).tensor  # 64x64

        hgd = [
            {"type": "conv2d", "depth": 64},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 32 x 32
            {"type": "conv2d", "depth": 128},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 16 x 16
            {"type": "conv2d", "depth": 256},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 8 x 8
            {"type": "conv2d", "depth": 512},
            {"type": "skip", "layer_num": 2},
            {"type": "pool", "pool": "max", "kernel": 2, "stride": 2},  # 4 x 4
            {"type": "conv2d", "depth": 512},
        ]

        with pt.defaults_scope(**self.pt_defaults_scope_value()):
            feature_map = hourglass(
                mid_tensor, hgd,
                net_type=self.options["hourglass_type"] if "hourglass_type" in self.options else None
            )

        return feature_map

    def bg_feature(self, image_tensor):
        with pt.defaults_scope(**self.pt_defaults_scope_value()):
            return (
                pt.wrap(image_tensor).
                conv2d(3, 32).max_pool(2, 2).   # 64
                conv2d(3, 64).max_pool(2, 2).   # 32
                conv2d(3, 128).max_pool(2, 2).  # 16
                conv2d(3, 256).max_pool(2, 2).  # 8
                conv2d(3, 256).max_pool(2, 2).  # 4
                conv2d(3, 512).max_pool(2, 2).  # 2
                conv2d(3, 512)
            )


