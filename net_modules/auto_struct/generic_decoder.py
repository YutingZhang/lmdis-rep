from abc import ABCMeta, abstractmethod
import tensorflow as tf
import prettytensor as pt
import zutils.tf_math_funcs as tmf
import net_modules.auto_struct.utils as asu
from net_modules import keypoints_2d
from net_modules.distribution_utils import reshape_extended_features
from zutils.py_utils import *
import zutils.pt_utils as ptu
import numpy as np


class Factory:

    __metaclass__ = ABCMeta

    structure_param_num = None

    feature_extended_num = 1

    def __init__(self, recon_dist_param_num=1, options=None):
        self.recon_dist_param_num = recon_dist_param_num
        self.options = options
        self.allow_overall = True

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
            self.patch_feature_dim = self.options["overall_feature_dim"]

    def __call__(self, input_tensor, condition_tensor=None, extra_inputs=None):
        im, _, mos = self.decode(input_tensor, condition_tensor=condition_tensor, extra_inputs=extra_inputs)
        return im, mos.extra_outputs

    def decode(self, input_tensor, condition_tensor=None, extra_inputs=None):
        """Create encoder network.
        """

        if extra_inputs is None:
            extra_inputs = dict()

        assert self.structure_param_num is not None, "structure_param_num is not defined"

        # module output strip
        mos = asu.ModuleOutputStrip()

        with tf.variable_scope("variational"):
            # latent to structure, patch, and overall
            structure_param_x, patch_features_x, overall_features_x = mos(
                self._latent2structure_patch_overall(latent_tensor=input_tensor))

            structure_param_x = mos(self.structure_postprocess(structure_param_x))

            actual_structure_param_num = tmf.get_shape(structure_param_x)[-1]
            the_param_factor = (actual_structure_param_num // self.structure_param_num)
            assert the_param_factor == self.feature_extended_num, "wrong dim for feature extension"
            structure_param_x, structure_param = reshape_extended_features(structure_param_x, the_param_factor)
            patch_features_x, patch_features = reshape_extended_features(patch_features_x, the_param_factor)
            overall_features_x, overall_features = reshape_extended_features(overall_features_x, the_param_factor)

            # store structure_param
            mos.extra_outputs["save"]["structure_param"] = structure_param

            # feed to discriminator (if needed)
            mos.extra_outputs["for_discriminator"]["structure_param"] = structure_param

        input_structure_param = structure_param
        input_patch_features = patch_features
        input_overall_features = overall_features

        if "override_structure_param" in extra_inputs and extra_inputs["override_structure_param"] is not None:
            input_structure_param = extra_inputs["override_structure_param"]
        if "override_patch_features" in extra_inputs and extra_inputs["override_patch_features"] is not None:
            input_patch_features = extra_inputs["override_patch_features"]
        if "override_overall_features" in extra_inputs and extra_inputs["override_overall_features"] is not None:
            input_overall_features = extra_inputs["override_overall_features"]

        input_structure_param0 = input_structure_param
        input_structure_param, input_overall_features, input_patch_features = \
            self.rotate_dominating_features_if_necessary(
                condition_tensor, input_structure_param, input_overall_features, input_patch_features
            )
        if input_structure_param is not input_structure_param0:
            # save actual used structure param
            mos.extra_outputs["save"]["structure_param0"] = mos.extra_outputs["save"]["structure_param"]
            mos.extra_outputs["save"]["structure_param"] = structure_param

        if self.input_feature_dim() is not None:
            input_patch_features = pt.wrap(input_patch_features).group_connected(
                self.input_feature_dim(),
                tie_groups=self.options["tie_patch_feature_spaces"]
                if "tie_patch_feature_spaces" in self.options else False
            )

        im, _ = self.decode_deterministic(
            input_structure_param,
            input_patch_features,
            input_overall_features,
            extra_inputs=extra_inputs,
            mos=mos
        )

        detailed_outputs = dict(
            structure_param_x=structure_param_x,
            structure_param=structure_param,
            patch_features_x=patch_features_x,
            patch_features=patch_features,
            overall_features_x=overall_features_x,
            overall_features=overall_features
        )

        return im, detailed_outputs, mos

    def decode_deterministic(
            self, structure_param, patch_features, overall_features, extra_inputs=None, mos=None,
            default_reuse=None
    ):

        if not self.allow_overall:
            assert overall_features is None, "Do not support overall_features"

        if mos is None:
            mos = asu.ModuleOutputStrip()

        with tf.variable_scope("deterministic", reuse=default_reuse):
            # build heatmap
            raw_heatmap_list = mos(self.structure2heatmap(structure_param, extra_inputs=extra_inputs))
            if not isinstance(raw_heatmap_list, (list, tuple)):
                raw_heatmap_list = [raw_heatmap_list]
            heatmap_list = list()
            for the_heatmap in raw_heatmap_list:
                heatmap_list.append(the_heatmap)
            mos.extra_outputs["save"]["heatmap"] = heatmap_list[0]
            heatmap = tf.concat(heatmap_list, axis=3)

            # build feature map (if needed)
            if patch_features is not None:
                # patch_features: [batch_size, struct_num, channels]
                # heatmap: [batch_size, h, w, struct_num]
                patch_features_e = tmf.expand_dims(patch_features, axis=1, ndims=2)
                feature_map_list = list()
                for the_heatmap in heatmap_list:
                    the_heatmap_e = tf.expand_dims(the_heatmap, axis=-1)
                    the_feature_map = tf.reduce_sum(patch_features_e * the_heatmap_e, axis=3)
                    feature_map_list.append(the_feature_map)
                feature_map = tf.concat(feature_map_list, axis=3)
                feature_map = tf.concat([heatmap, feature_map], axis=3)
            else:
                feature_map = heatmap

            im = mos(self.feature2image_with_overall(feature_map, overall_features))
            im = call_func_with_ignored_args(
                self.post_image_reconstruction, im, extra_inputs=extra_inputs
            )

        return im, mos

    def structure_postprocess(self, structure_param):
        return structure_param, None

    def _latent2structure_patch_overall(self, latent_tensor):
        a = self.__latent2structure_patch_overall(latent_tensor)
        assert isinstance(a, tuple), "wrong output type"
        assert a[0] is not None, "it seems the latent to structure mapping is not defined"
        if len(a) == 4:
            return a
        elif len(a) == 3:
            return a + (None,)
        else:
            raise ValueError("wrong number of outputs")

    # exactly one of the following four methods should be overridden -------------
    def __latent2structure_patch_overall(self, latent_tensor):
        struct_patch_overall = self.latent2structure_patch_overall(latent_tensor)
        struct_patch = self.latent2structure_patch(latent_tensor)
        struct_overall = self.latent2structure_overall(latent_tensor)
        struct_only = self.latent2structure(latent_tensor)
        user_def_embedding = list(
            filter(lambda x: x is not None, [struct_patch_overall, struct_patch, struct_overall, struct_only]))

        if not user_def_embedding:
            struct_patch_overall = self.latent2structure_patch_overall_generic(latent_tensor)
            if struct_patch_overall is not None:
                user_def_embedding.append(struct_patch_overall)
        assert len(user_def_embedding) == 1, \
            "exactly one of latent2structure_* should be override"

        def wrap_from_1(a):
            if isinstance(a, tuple):
                if len(a) == 1:
                    return a[0], None, None, None
                elif len(a) == 2:
                    return a[0], None, None, a[1]
                else:
                    raise ValueError("wrong number of outputs")
            else:
                return a, None, None, None

        def wrap_from_2(a, at_third=False):
            assert isinstance(a, tuple), "wrong output type"
            if len(a) == 2:
                return (a[0],) + ((None, a[1]) if at_third else (a[1], None)) + (None,)
            elif len(a) == 3:
                return (a[0],) + ((None, a[1]) if at_third else (a[1], None)) + (a[2],)
            else:
                raise ValueError("wrong number of outputs")

        def wrap_from_3(a):
            assert isinstance(a, tuple), "wrong output type"
            if len(a) == 3:
                return a + (None,)
            else:
                return a

        if struct_patch_overall is not None:
            return wrap_from_3(struct_patch_overall)
        if struct_patch is not None:
            return wrap_from_2(struct_patch, at_third=False)
        elif struct_overall is not None:
            return wrap_from_2(struct_overall, at_third=True)
        elif struct_only is not None:
            return wrap_from_1(struct_only)
        else:
            raise ValueError("internal errors: did not find any actual definition")

    def rotate_dominating_features_if_necessary(self, condition_tensor, structure_param, *args):

        if "keypoint_diff_factor" in self.options:
            self.options["structure_diff_factor"] = self.options["keypoint_diff_factor"]
        if "keypoint_rotating_indexes" in self.options:
            self.options["structure_rotating_indexes"] = self.options["keypoint_rotating_indexes"]

        sample_id_for_dominating_feature = Factory._sample_id_for_dominating_feature_from_condition(
            condition_tensor
        )
        outputs = list(args)
        if ptu.default_phase() != pt.Phase.test or sample_id_for_dominating_feature is None:
            outputs = (structure_param,) + tuple(outputs)
            return outputs

        for i in range(len(outputs)):
            if outputs[i] is not None:
                outputs[i] = tf.tile(
                    tf.expand_dims(outputs[i][sample_id_for_dominating_feature], axis=0),
                    [tmf.get_shape(outputs[i])[0]] + [1] * (len(tmf.get_shape(outputs[i]))-1)
                )

        batch_size = tmf.get_shape(structure_param)[0]
        num_structure = tmf.get_shape(structure_param)[1]
        use_selected_index = "structure_rotating_indexes" in self.options and self.options["structure_rotating_indexes"]
        if use_selected_index:
            chosen_dim_idxb = np.zeros(num_structure, np.bool_)
            chosen_dim_idxb[self.options["structure_rotating_indexes"]] = np.True_
            chosen_ndim = np.sum(chosen_dim_idxb)
        else:
            chosen_dim_idxb = np.ones(num_structure, np.bool_)
            chosen_ndim = num_structure
        structure_tile_vec = [batch_size] + [1]*(len(tmf.get_shape(structure_param))-1)
        chosen_dim_idxb = np.expand_dims(chosen_dim_idxb, axis=0)
        for _ in range(len(tmf.get_shape(structure_param))-2):
            chosen_dim_idxb = np.expand_dims(chosen_dim_idxb, axis=-1)
        chosen_dim_idxbi = chosen_dim_idxb.astype(np.int32)
        chosen_dim_idxb_x = np.tile(chosen_dim_idxb, structure_tile_vec)

        ref_structure_param = tf.expand_dims(structure_param[sample_id_for_dominating_feature], axis=0)
        if "structure_diff_factor" in self.options:
            structure_diff_factor = self.options["structure_diff_factor"] \
                if self.options["structure_diff_factor"] is not None else 1
            if structure_diff_factor != 1:
                structure_param = \
                    ref_structure_param * (1-structure_diff_factor) + structure_param * structure_diff_factor
        if use_selected_index:
            structure_param = chosen_dim_idxbi * structure_param + (1-chosen_dim_idxbi) * ref_structure_param

        outputs = (structure_param,) + tuple(outputs)
        return outputs

    @staticmethod
    def _sample_id_for_dominating_feature_from_condition(condition_tensor):
        if condition_tensor is None:
            return None
        for c in condition_tensor:
            if c["type"] == "boolmask_for_dominating_sample_in_batch_rotating":
                the_id = tf.argmax(tf.to_int32(c["value"]), 0)
                the_id = tf.reshape(the_id, [])
                return the_id
        return None

    def latent2structure_patch_overall_generic(self, latent_tensor):
        # automatic being overrided if one of the following four function got overrided
        return None

    def latent2structure_patch_overall(self, latent_tensor):
        return None

    def latent2structure_overall(self, latent_tensor):
        return None

    def latent2structure_patch(self, latent_tensor):
        return None

    def latent2structure(self, latent_tensor):
        return None
    # ------------------------------------------------------------------------------

    @abstractmethod
    def structure2heatmap(self, structure_param, extra_inputs=None):
        pass

    # override one of the following -----------------------------------------------
    def feature2image_with_overall(self, feature_map, overall_feature):
        return self.feature2image(feature_map)

    def feature2image(self, feature_map):
        raise ValueError("feature2image is not defined")
        return None

    def input_feature_dim(self):
        return None

    def post_image_reconstruction(self, im, extra_inputs=None):
        return im
