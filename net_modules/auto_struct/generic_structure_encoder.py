from abc import ABCMeta, abstractmethod
import tensorflow as tf
import prettytensor as pt
import zutils.tf_math_funcs as tmf
import zutils.pt_utils as ptu
import net_modules.auto_struct.utils as asu
from net_modules import keypoints_2d
import math
from zutils.py_utils import *
import zutils.tf_graph_utils as tgu

import collections


class Factory:
    __metaclass__ = ABCMeta

    def __init__(self, output_channels, options):
        """
        :param output_channels: output_channels for the encoding net
        """
        self.output_channels = output_channels
        self.options = options

        self.structure_as_final_class = True
        self.target_input_size = None

        self.stop_gradient_at_latent_for_structure = False

    def __call__(self, input_tensor, condition_tensor=None, extra_inputs=None):
        _, _, structure_latent, mos = self.structure_encode(
            input_tensor, condition_tensor=condition_tensor, extra_inputs=extra_inputs)
        latent_tensor = structure_latent
        assert self.output_channels == tmf.get_shape(latent_tensor)[1], \
            "wrong output_channels"
        return structure_latent, mos.extra_outputs

    def input_to_heatmap_overall(self, input_tensor, mos):
        # compute shared features
        overall_feature = mos(self.image2sharedfeature(input_tensor))
        # compute raw heatmap
        raw_heatmap = mos(self.image2heatmap(overall_feature))
        if "heatmap_extra" in mos.extra_outputs:
            heatmap_extra = mos.extra_outputs["heatmap_extra"]
        else:
            heatmap_extra = None
        raw_heatmap = mos(call_func_with_ignored_args(
            self.heatmap_postprocess, raw_heatmap, image_tensor=input_tensor, heatmap_extra=heatmap_extra))
        if "heatmap_extra" in mos.extra_outputs:
            heatmap_extra = mos.extra_outputs["heatmap_extra"]
        else:
            heatmap_extra = None
        # normalize heatmap
        heatmap = tf.nn.softmax(raw_heatmap)
        heatmap = mos(call_func_with_ignored_args(
            self.heatmap_postpostprocess, heatmap, image_tensor=input_tensor, heatmap_extra=heatmap_extra
        ))
        return heatmap, overall_feature

    def input_to_heatmap(self, input_tensor, mos, **kwargs):
        heatmap, _ = self.input_to_heatmap_overall(input_tensor, mos, **kwargs)
        return heatmap

    def structure_encode(self, input_tensor, condition_tensor=None, extra_inputs=None):
        if "freeze_encoded_structure" in self.options and rbool(self.options["freeze_encoded_structure"]):
            with pt.defaults_scope(phase=pt.Phase.test):
                return self.structure_encode_(input_tensor, condition_tensor, extra_inputs)
        else:
            return self.structure_encode_(input_tensor, condition_tensor, extra_inputs)

    def structure_encode_(self, input_tensor, condition_tensor=None, extra_inputs=None):
        """Create encoder network.
        """

        input_tensor = self.pad_input_tensor(input_tensor)

        # module output strip

        mos = asu.ModuleOutputStrip()
        mos.extra_outputs["discriminator_remark"] = dict(
            generator_aux_loss=[]
        )

        deterministic_collection = tgu.SubgraphCollectionSnapshots()
        deterministic_collection.sub_snapshot("_old")
        with tf.variable_scope("deterministic"), tf.variable_scope("structure"):

            # augment images (if needed)
            main_batch_size = tmf.get_shape(input_tensor)[0]

            input_tensor_x, aug_cache = mos(self.augment_images(input_tensor))
            network_predefined = ("network_predefined" in aug_cache) and aug_cache["network_predefined"]

        aug_cache["main_batch_size"] = main_batch_size
        mos.extra_outputs["aug_cache"] = aug_cache

        with tf.variable_scope("deterministic"):
            with tf.variable_scope("structure", reuse=True if network_predefined else None):
                heatmap, overall_feature = self.input_to_heatmap_overall(input_tensor_x, mos)
                structure_pack = mos(self.heatmap2structure(heatmap))

            with tf.variable_scope("structure"):
                # postprocess structure
                structure_param_x = mos(call_func_with_ignored_args(
                    self.heatmap2structure_poststep, structure_pack, image_tensor=input_tensor_x
                ))
                # clean up augmented data
                structure_param = mos(call_func_with_ignored_args(
                    self.cleanup_augmentation_structure, structure_param_x, aug_cache=aug_cache,
                    condition_tensor=condition_tensor
                ))

        with tf.variable_scope("deterministic"), tf.variable_scope("structure"):
            mos.extra_outputs["save"]["heatmap"] = heatmap[:main_batch_size]

            # entropy loss to encourage heatmap separation across different channels
            if "heatmap_separation_loss_weight" in self.options and \
                    rbool(self.options["heatmap_separation_loss_weight"]):
                total_heatmap_entropy = keypoints_2d.keypoint_map_depth_entropy_with_real_bg(heatmap)
                separation_loss = total_heatmap_entropy * self.options["heatmap_separation_loss_weight"]
                separation_loss.disp_name = "separation"
                tgu.add_to_aux_loss(separation_loss)

            # register structure_param for storing
            mos.extra_outputs["save"]["structure_param"] = structure_param
            mos.extra_outputs["for_decoder"]["structure_param"] = structure_param

            # structure_param matching
            if extra_inputs is not None and "structure_param" in extra_inputs and \
                    "structure_detection_loss_weight" in self.options and \
                    rbool(self.options["structure_detection_loss_weight"]):
                structure_param_dist = self.structure_param_distance(
                    extra_inputs["structure_param"], tf.stop_gradient(structure_param))
                structure_detection_loss = \
                    self.options["structure_detection_loss_weight"] * tf.reduce_mean(structure_param_dist, axis=0)
                structure_detection_loss.disp_name = "struct_detection"
                tgu.add_to_aux_loss(structure_detection_loss)
                mos.extra_outputs["discriminator_remark"]["generator_aux_loss"].append(structure_detection_loss)

        deterministic_collection.sub_snapshot("structure_deterministic")
        encoded_structure_vars = deterministic_collection["structure_deterministic"].get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)
        if "freeze_encoded_structure" in self.options and rbool(self.options["freeze_encoded_structure"]):
            tgu.add_to_freeze_collection(encoded_structure_vars)
        if "encoded_structure_lr_mult" in self.options and rbool(self.options["encoded_structure_lr_mult"]):
            for v in encoded_structure_vars:
                v.lr_mult = self.options["encoded_structure_lr_mult"]

        with tf.variable_scope("variational"), tf.variable_scope("structure"):
            structure_latent = mos(self.structure2latent(structure_param))

        if self.structure_as_final_class:
            with tf.variable_scope("deterministic"):
                # use the main batch only
                heatmap = heatmap[:main_batch_size]
                overall_feature = overall_feature[:main_batch_size]

        return overall_feature, heatmap, structure_latent, mos

    def augment_images(self, image_tensor):
        return image_tensor

    def cleanup_augmentation_structure(self, structure_param, aug_cache, condition_tensor=None):
        return structure_param

    def image2sharedfeature(self, image_tensor):
        return image_tensor

    @abstractmethod
    def image2heatmap(self, image_tensor):
        return None

    def heatmap_postprocess(self, heatmap):
        return heatmap

    def heatmap_postpostprocess(self, heatmap):
        return heatmap

    def heatmap2structure_poststep(self, structure_pack):
        return structure_pack

    @abstractmethod
    def heatmap2structure(self, heatmap_tensor):
        return None

    def structure2latent(self, structure_tensor):
        # simply copy the structure as latent
        input_shape = tmf.get_shape(structure_tensor)
        latent_tensor = tf.reshape(structure_tensor, [input_shape[0], -1])
        return latent_tensor

    def structure_param2euclidean(self, structure_param):
        return structure_param

    def structure_param_distance(self, p1, p2):
        batch_size = tmf.get_shape(p1)[0]
        r1 = self.structure_param2euclidean(p1)
        r2 = self.structure_param2euclidean(p2)
        r1 = tf.reshape(r1, [batch_size, -1])
        r2 = tf.reshape(r2, [batch_size, -1])
        return tf.reduce_sum(tf.square(r2-r1), axis=1)

    def pad_input_tensor(self, input_tensor):

        if self.target_input_size is None:
            return input_tensor

        if (
            isinstance(self.target_input_size, collections.Iterable) and
            isinstance(self.target_input_size, collections.Sized)
        ):
            assert len(self.target_input_size) == 2, "wrong target_input_size"
            final_input_size = self.target_input_size
        else:
            final_input_size = [self.target_input_size] * 2

        init_input_size = tmf.get_shape(input_tensor)[1:3]

        assert math.isclose(final_input_size[0]/init_input_size[0], final_input_size[1]/init_input_size[1]), \
            "enlarge ratio should be the same (for the simplicity of other implementation)"

        assert final_input_size[0] >= init_input_size[0] and final_input_size[1] >= init_input_size[1], \
            "target input size should not be smaller the actual input size"

        if init_input_size[0] == final_input_size[0] and init_input_size[1] == final_input_size[1]:
            return input_tensor
        else:
            the_pad_y_begin = (final_input_size[0] - init_input_size[0]) // 2
            the_pad_x_begin = (final_input_size[1] - init_input_size[1]) // 2
            the_padding = [
                [0, 0],
                [the_pad_y_begin, final_input_size[0] - init_input_size[0] - the_pad_y_begin],
                [the_pad_x_begin, final_input_size[1] - init_input_size[1] - the_pad_x_begin],
                [0] * 2,
            ]
            paded_input_tensor = tmf.pad(
                tensor=input_tensor, paddings=the_padding, mode="MEAN_EDGE",
                geometric_axis=[1, 2]
            )
            return paded_input_tensor

