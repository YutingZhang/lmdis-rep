from abc import ABCMeta, abstractmethod
from net_modules import keypoints_2d
import tensorflow as tf
import zutils.tf_math_funcs as tmf
import numpy as np
import zutils.pt_utils as ptu
import prettytensor as pt
import math

import warnings

from net_modules.auto_struct.generic_decoder import Factory as BaseFactory

from net_modules.auto_struct.keypoint_encoder import Factory as EncoderFactory


class BasicFactory(BaseFactory):

    structure_param_num = EncoderFactory.structure_param_num

    def __init__(self, recon_dist_param_num=1, options=None):
        super().__init__(recon_dist_param_num, options)
        self.keypoint_init()

    def keypoint_init(self):

        self.base_gaussian_stddev = keypoints_2d.gaussian_2d_base_stddev
        if "base_gaussian_stddev" in self.options and self.options["base_gaussian_stddev"] is not None:
            self.base_gaussian_stddev = self.options["base_gaussian_stddev"]

        self.use_background_feature = "background_as_patch" not in self.options or self.options["background_as_patch"]

    @abstractmethod
    def image_size(self):
        raise ValueError("Must specify the image size.")
        return None, None

    def structure2heatmap(self, structure_param, extra_inputs=None):

        if extra_inputs is None:
            extra_inputs = dict()

        h, w = self.image_size()

        keypoint_param_raw = structure_param

        if "heatmap_stddev_for_patch_features" in extra_inputs and \
                extra_inputs["heatmap_stddev_for_patch_features"] is not None:
            the_base_gaussian_stddev = extra_inputs["heatmap_stddev_for_patch_features"]
        else:
            the_base_gaussian_stddev = tf.ones_like(keypoint_param_raw)*self.base_gaussian_stddev

        def param2heatmap(std_scale=1):
            if std_scale == 1:
                keypoint_param = keypoint_param_raw
            else:
                param_dim = tmf.get_shape(keypoint_param_raw)[2]
                if param_dim == 2:
                    keypoint_param = tf.concat([
                        keypoint_param_raw, the_base_gaussian_stddev*std_scale
                    ], axis=2)
                elif param_dim == 3:
                    keypoint_param = tf.concat([
                        keypoint_param_raw[:, :, :2],
                        keypoint_param_raw[:, :, 2:3]*std_scale
                    ], axis=2)
                else:
                    keypoint_param = tf.concat([
                        keypoint_param_raw[:, :, :2],
                        keypoint_param_raw[:, :, 2:4]*std_scale
                    ], axis=2)
                    if param_dim==5:
                        keypoint_param = tf.concat([
                            keypoint_param,
                            keypoint_param_raw[:, :, 4:5]
                        ], axis=2)
            keypoint_map = keypoints_2d.gaussian_coordinate_to_keypoint_map(
                keypoint_param, h, w
            )
            keypoint_map_with_bg = tf.concat(
                [keypoint_map, tf.ones_like(keypoint_map[:, :, :, 0:1]) * (1. / (h * w))], axis=3
            )
            keypoint_map_with_bg /= tf.reduce_sum(keypoint_map_with_bg, axis=3, keep_dims=True)
            return keypoint_map_with_bg

        if "keypoint_decoding_heatmap_levels" not in self.options or \
                self.options["keypoint_decoding_heatmap_levels"] == 1:
            return param2heatmap()
        else:
            assert "keypoint_decoding_heatmap_level_base" in self.options, \
                "keypoint_decoding_heatmap_level_base must be specified"
            b = self.options["keypoint_decoding_heatmap_level_base"]
            s = 1.
            keypoint_map_list = list()
            for i in range(self.options["keypoint_decoding_heatmap_levels"]):
                keypoint_map_list.append(param2heatmap(std_scale=s))
                s /= b
            return keypoint_map_list

    def structure_param2euclidean(self, structure_param):
        return keypoints_2d.gaussian2dparam_to_recon_code(structure_param)

    def post_image_reconstruction(self, im, extra_inputs=None):
        extra_outputs = dict()
        extra_outputs["save"] = dict()
        return im


class Factory(BasicFactory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def latent2structure_patch_overall_generic(self, latent_tensor):

        keypoint_num = self.options["keypoint_num"]

        batch_size = tmf.get_shape(latent_tensor)[0]
        total_dim = tmf.get_shape(latent_tensor)[1]

        cur_dim = 0
        keypoint_param_dim = self.structure_param_num * keypoint_num
        keypoint_tensor = latent_tensor[:, :keypoint_param_dim]
        keypoint_param = tf.reshape(keypoint_tensor, [batch_size, keypoint_num, -1])
        cur_dim += keypoint_param_dim

        if self.patch_feature_dim is not None and self.patch_feature_dim > 0:
            all_patch_feat_dims = (keypoint_num+1)*self.patch_feature_dim
            patch_tensor = latent_tensor[:, keypoint_param_dim:keypoint_param_dim+all_patch_feat_dims]
            patch_features = tf.reshape(patch_tensor, [
                batch_size, keypoint_num + (1 if self.use_background_feature else 0),
                self.patch_feature_dim
            ])
            cur_dim += all_patch_feat_dims
        else:
            patch_features = None

        if total_dim > cur_dim:
            overall_features = latent_tensor[:, cur_dim:]
            if self.overall_feature_dim is None or self.overall_feature_dim+cur_dim<total_dim:
                warnings.warn("mismatch overall feature dim specification")
        else:
            overall_features = None

        if not self.use_background_feature:
            # set background features to zeros
            patch_features = tf.concat([
                patch_features[:, :-1, :], tf.zeros_like(patch_features[:, -1:, :])
            ], axis=1)

        return keypoint_param, patch_features, overall_features, None
