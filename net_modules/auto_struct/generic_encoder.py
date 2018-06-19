from abc import ABCMeta, abstractmethod
from net_modules.auto_struct.generic_structure_encoder import Factory as BaseFactory
import tensorflow as tf
import prettytensor as pt
import zutils.tf_math_funcs as tmf
import net_modules.auto_struct.utils as asu
from zutils.py_utils import dummy_class_for_with


class Factory(BaseFactory):

    # __metaclass__ = ABCMeta

    def __init__(self, output_channels, options):
        """
        :param output_channels: output_channels for the encoding net
        """
        super().__init__(output_channels, options)

        self.structure_as_final_class = False

        self.stop_gradient_at_latent_for_patch = False
        self.stop_gradient_at_latent_for_overall = False

        # patch feature dim (utilities for inherent class)
        if hasattr(self, "default_patch_feature_dim"):
            self.patch_feature_dim = self.default_patch_feature_dim
        else:
            self.patch_feature_dim = None
        if "patch_feature_dim" in self.options:
            self.patch_feature_dim = self.options["patch_feature_dim"]

        # overall feature dim (utilities for inherent class)
        if hasattr(self, "default_overall_feature_dim"):
            self.overall_feature_dim = self.default_overall_feature_dim
        else:
            self.overall_feature_dim = None
        if "overall_feature_dim" in self.options:
            self.overall_feature_dim = self.options["overall_feature_dim"]

    def __call__(self, input_tensor, condition_tensor=None, extra_inputs=None):
        """Create encoder network.
        """

        latent_tensor, mos = self.patch_structure_overall_encode(
            input_tensor, condition_tensor=condition_tensor, extra_inputs=extra_inputs
        )
        assert self.output_channels == tmf.get_shape(latent_tensor)[1], \
            "wrong output_channels"
        return latent_tensor, mos.extra_outputs

    def patch_structure_overall_encode(self, input_tensor, condition_tensor=None, extra_inputs=None):
        """Create encoder network.
        """

        # compute structures
        overall_feature, heatmap, structure_latent, mos = self.structure_encode(
            input_tensor, condition_tensor=condition_tensor, extra_inputs=extra_inputs)

        if "heatmap_for_patch_features" in mos.extra_outputs \
                and mos.extra_outputs["heatmap_for_patch_features"] is not None:
            heatmap_for_patch_features = mos.extra_outputs["heatmap_for_patch_features"]
        else:
            heatmap_for_patch_features = heatmap

        aug_cache = mos.extra_outputs["aug_cache"]
        main_batch_size = aug_cache["main_batch_size"]

        with tf.variable_scope("deterministic"), tf.variable_scope("feature"):
            # compute patch features
            feature_map = mos(self.image2feature(overall_feature))

            patch_features = None
            if feature_map is not None:
                # pool features
                batch_size, h, w, num_struct = tmf.get_shape(heatmap_for_patch_features)
                feature_channels = tmf.get_shape(feature_map)[-1]
                heatmap_e = tf.reshape(
                    heatmap_for_patch_features, [batch_size, h*w, num_struct, 1])
                feature_map_e = tf.reshape(
                    feature_map, [batch_size, h*w, 1, feature_channels])

                patch_features = tf.reduce_sum(feature_map_e * heatmap_e, axis=1) / \
                    tf.reduce_sum(heatmap_e, axis=1)  # [batch_size, struct_num, feature_channels]

                # if tmf.get_shape(patch_features)[2] != self.patch_feature_dim:
                # always add an independent feature space
                if hasattr(self, "pt_defaults_scope_value"):
                    pt_scope = pt.defaults_scope(**self.pt_defaults_scope_value())
                else:
                    pt_scope = dummy_class_for_with()
                with pt_scope:
                    patch_features = pt.wrap(patch_features).group_connected(
                        self.patch_feature_dim, activation_fn=None,
                        tie_groups=self.options["tie_patch_feature_spaces"]
                        if "tie_patch_feature_spaces" in self.options else False
                    )

        with tf.variable_scope("deterministic"):
            # use the main batch only
            heatmap = heatmap[:main_batch_size]
            overall_feature = overall_feature[:main_batch_size]
            if patch_features is not None:
                patch_features = mos(self.cleanup_augmentation_patchfeatures(patch_features, aug_cache))

        mos.extra_outputs["for_decoder"]["patch_features"] = patch_features

        with tf.variable_scope("variational"):  # use the scope name for backward compitability
            with tf.variable_scope("feature"):
                # compute patch latent
                patch_latent = None
                if patch_features is not None:
                    if self.stop_gradient_at_latent_for_patch:
                        patch_features_1 = tf.stop_gradient(patch_features)
                    else:
                        patch_features_1 = patch_features
                    patch_latent = mos(self.feature2latent(patch_features_1))

                # compute overall latent
                if self.stop_gradient_at_latent_for_overall:
                    overall_feature_1 = tf.stop_gradient(overall_feature)
                else:
                    overall_feature_1 = overall_feature
                overall_latent = self.overall2latent(overall_feature_1)

            with tf.variable_scope("both"):
                latent_tensor = mos(self._fuse_structure_patch_overall(
                    structure_latent, patch_latent, overall_latent
                ))

        return latent_tensor, mos

    def cleanup_augmentation_patchfeatures(self, patch_features, aug_cache):
        return patch_features[:aug_cache["main_batch_size"]]

    def image2feature(self, image_tensor):
        return None

    def feature2latent(self, patch_features):
        batch_size, struct_num, feature_channels = tmf.get_shape(patch_features)
        return tf.reshape(patch_features, [batch_size, struct_num * feature_channels])

    def overall2latent(self, image_features):
        return None

    def _fuse_structure_patch_overall(self, structure_latent, patch_latent, overall_latent):
        assert structure_latent is not None, "structure latent should not be None"
        if patch_latent is None and overall_latent is None:
            return self.fuse_structure(structure_latent)
        elif patch_latent is None:
            return self.fuse_structure_overall(structure_latent, overall_latent)
        elif overall_latent is None:
            return self.fuse_structure_patch(structure_latent, patch_latent)
        else:
            return self.fuse_structure_patch_overall(structure_latent, patch_latent, overall_latent)

    # override one of the following three methods based on which features are used --------------------

    def fuse_structure(self, structure_latent):
        return structure_latent

    def fuse_structure_patch(self, structure_latent, patch_latent):
        return tf.concat([structure_latent, patch_latent], axis=-1)

    def fuse_structure_overall(self, structure_latent, overall_latent):
        return tf.concat([structure_latent, overall_latent], axis=-1)

    def fuse_structure_patch_overall(self, structure_latent, patch_latent, overall_latent):
        return tf.concat([structure_latent, patch_latent, overall_latent], axis=-1)
    # -------------------------------------------------------------------------------------------------

