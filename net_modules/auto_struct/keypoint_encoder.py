from net_modules import keypoints_2d
import tensorflow as tf
import zutils.tf_math_funcs as tmf
import numpy as np
import zutils.pt_utils as ptu
import net_modules.auto_struct.utils as asu
import prettytensor as pt
import math
from zutils.py_utils import *
import zutils.tf_graph_utils as tgu


from net_modules.auto_struct.generic_encoder import Factory as BaseFactory


class BasicFactory(BaseFactory):

    structure_param_num = 2

    def __init__(self, output_channels, options):
        """
        :param output_channels: output_channels for the encoding net
        """
        super().__init__(output_channels, options)
        self.keypoint_init()

    def keypoint_init(self):

        self.input_size = None

        self.base_gaussian_stddev = keypoints_2d.gaussian_2d_base_stddev
        if "base_gaussian_stddev" in self.options and self.options["base_gaussian_stddev"] is not None:
            self.base_gaussian_stddev = self.options["base_gaussian_stddev"]

        self.enable_random_transform = True

        if "lm_tps_probability" in self.options:
            self.lm_tps_probability = self.options["lm_tps_probability"]
        else:
            self.lm_tps_probability = 0.3

    def use_random_transform(self):
        # random for transformation for train phase only
        return (
            self.enable_random_transform and
            ptu.default_phase() == pt.Phase.train and
            "keypoint_transform_loss_weight" in self.options and
            rbool(self.options["keypoint_transform_loss_weight"]) and
            not ("freeze_encoded_structure" in self.options and rbool(self.options["freeze_encoded_structure"]))
        )

    def image_size(self, img_h, img_w):

        if self.target_input_size is None:
            return img_h, img_w, img_h, img_w

        if isinstance(self.input_size, (list, tuple)):
            actual_h = self.input_size[0]
            actual_w = self.input_size[1]
        else:
            actual_h = self.input_size
            actual_w = self.input_size
        if isinstance(self.target_input_size, (list, tuple)):
            full_h = self.target_input_size[0]
            full_w = self.target_input_size[1]
        else:
            full_h = self.target_input_size
            full_w = self.target_input_size
        return actual_h, actual_w, full_h, full_w

    def augment_images(self, image_tensor):

        if not hasattr(self, "target_size"):
            if hasattr(self, "input_size") and self.input_size is not None:
                self.target_size = self.input_size * 2

        actual_h, actual_w, full_h, full_w = \
            self.image_size(tmf.get_shape(image_tensor)[0], tmf.get_shape(image_tensor)[1])

        # random data augmentation for transformation invariance

        aug_cache = dict()
        aug_cache["original_image"] = image_tensor

        if not self.use_random_transform():
            return image_tensor, aug_cache, None

        batch_size = tmf.get_shape(image_tensor)[0]

        # get the landmarks using current model
        mos_tmp = asu.ModuleOutputStrip()
        with tgu.EnableAuxLoss(False):
            main_heatmap = self.input_to_heatmap(image_tensor, mos_tmp)
            main_keypoint_param = self.heatmap2structure_basic(main_heatmap)
        main_keypoint_param = main_keypoint_param[:, :, :2]
        del mos_tmp
        aug_cache["network_predefined"] = True  # in the parent function reuse=True for network definition

        with tf.variable_scope("transform_invariance"):

            h = tmf.get_shape(image_tensor)[1]
            w = tmf.get_shape(image_tensor)[2]
            im = image_tensor
            im_shape = tmf.get_shape(im)

            # ---- RANDOM LANDMARK TPS TRANSFORM ----
            lm_n_points = tmf.get_shape(main_keypoint_param)[1]
            lm_rand_pt_std = 0.05 #0.1
            lm_tps_cp = tf.random_normal(shape=[batch_size, lm_n_points, 2], stddev=lm_rand_pt_std)
            lm_tps_cp *= np.sqrt(np.reshape([full_w/full_h, full_h/full_w], [1, 1, 2]))
            # remark: y,x: y enlarge normalized coordinate according to aspect ratio, x shrink normalized coordinate
            lm_tps_fp = self.coordinate_to_stn(main_keypoint_param, aspect_ratio=full_w/full_h)
            lm_tps_fp = tf.stop_gradient(lm_tps_fp)

            im_t_1 = pt.wrap(im).spatial_transformer_tps(
                None, None, lm_tps_cp, out_size=[h, w], fp_more=lm_tps_fp
            )
            im_t_1 = tf.reshape(im_t_1, im_shape)

            aug_cache["lm_tps"] = dict()
            aug_cache["lm_tps"]["transform"] = lm_tps_cp
            aug_cache["lm_tps"]["control_points"] = lm_tps_fp
            aug_cache["lm_tps"]["num_points"] = lm_n_points

            # ---- RANDOM TPS TRANSFORM ----
            n_points = 7
            rand_pt_std = 0.1 # 0.2
            tps_transform = tf.random_normal(shape=[batch_size, n_points*n_points, 2], stddev=rand_pt_std)

            im_t_2 = pt.wrap(im).spatial_transformer_tps(
                n_points, n_points,
                tps_transform,
                out_size=[h, w],
            )
            im_t_2 = tf.reshape(im_t_2, im_shape)

            aug_cache["tps"] = dict()
            aug_cache["tps"]["transform"] = tps_transform
            aug_cache["tps"]["num_points"] = n_points

            # -------------- SELECT RANDOM TPS --------------------
            global_step = tf.train.get_global_step()
            lm_tps_step_lower = 5000
            lm_tps_step_upper = 10000
            lm_tps_random_upper_th = self.lm_tps_probability
            lm_tps_random_th = tf.where(
                global_step <= lm_tps_step_lower, tf.constant(0, dtype=tf.float32),
                tf.where(
                    global_step > lm_tps_step_upper, tf.constant(1, dtype=tf.float32),
                    tf.to_float(global_step-lm_tps_step_lower)/(lm_tps_step_upper-lm_tps_step_lower)
                ) * lm_tps_random_upper_th
            )
            use_lm_tps = tf.random_uniform([batch_size]) < lm_tps_random_th
            use_lm_tps = tf.zeros_like(use_lm_tps)
            im_t = tf.where(
                tf.tile(tmf.expand_dims(use_lm_tps, axis=-1, ndims=3), [1] + im_shape[1:]),
                im_t_1, im_t_2
            )
            aug_cache["use_lm_tps"] = use_lm_tps

            # ---- RANDOM SIMILARITY TRANSFORM ----
            # generate random transformation and generate the image
            trans_range = np.array([-0.15, 0.15])  # translation
            rotation_std = 10   # degree
            scale_std = 1.25   # scale

            # canonicalize parameter range
            rotation_std = rotation_std/180 * np.pi
            scale_std = np.log(scale_std)
            trans_range = trans_range * 2.  # spatial transformer use [-1, 1] for the coordinates

            # generate random transformation
            rand_base_t = tf.random_uniform(shape=[batch_size, 2, 1])
            rand_trans = rand_base_t*(trans_range[1]-trans_range[0]) + trans_range[0]   # trans x, y
            rand_rotation = tf.random_normal(shape=[batch_size, 1, 1]) * rotation_std
            rand_scale = tf.exp(tf.random_normal(shape=[batch_size, 1, 1]) * scale_std)

            if "keypoint_random_horizontal_mirroring" in self.options and \
                    self.options["keypoint_random_horizontal_mirroring"]:
                horizontal_sign = tf.to_float(tf.random_uniform([batch_size, 1, 1]) > 0.5)
            else:
                horizontal_sign = 1.
            if "keypoint_random_vertical_mirroring" in self.options and \
                    self.options["keypoint_random_vertical_mirroring"]:
                vertical_sign = tf.to_float(tf.random_uniform([batch_size, 1], 1) > 0.5)
            else:
                vertical_sign = 1.

            # concatenate parameters
            rand_cos = tf.cos(rand_rotation)
            rand_sin = tf.sin(rand_rotation)
            rand_rot_matrix = tf.concat(
                [
                    tf.concat([rand_cos, rand_sin], axis=1)*horizontal_sign,
                    tf.concat([-rand_sin, rand_cos], axis=1)*vertical_sign,
                ], axis=2)
            rand_sim_matrix = tf.concat(
                [rand_scale*rand_rot_matrix, rand_trans],
                axis=2
            )
            transform = rand_sim_matrix

            im_t = pt.wrap(im_t).spatial_transformer(
                tf.reshape(transform, [batch_size, 6]), out_size=im_shape[1:3]
            )
            im_t = tf.reshape(im_t, im_shape)

            aug_cache["sim_transform"] = transform

            # fuse converted images
            im_a = tf.concat([im, im_t], axis=0)

        return im_a, aug_cache, None

    def heatmap_postprocess(self, heatmap):
        extra_outputs = dict()
        extra_outputs["heatmap_extra"] = dict()
        heatmap_ch = tmf.get_shape(heatmap)[3]
        expected_channels = self.options["keypoint_num"] + 1
        if heatmap_ch != self.options["keypoint_num"] + 1:
            extra_outputs["heatmap_extra"]["feature"] = heatmap
            if hasattr(self, "pt_defaults_scope_value"):
                pt_scope = pt.defaults_scope(**self.pt_defaults_scope_value())
            else:
                pt_scope = dummy_class_for_with()
            with pt_scope:
                heatmap = pt.wrap(heatmap).conv2d(1, expected_channels, activation_fn=None)
        return heatmap, extra_outputs

    def heatmap_postpostprocess(self, heatmap, image_tensor=None, heatmap_extra=None):
        extra_outputs = dict()
        extra_outputs["for_decoder"] = dict()
        extra_outputs["save"] = dict()

        return heatmap, extra_outputs

    def heatmap2structure_internal(self, heatmap_tensor):

        keypoint_map = heatmap_tensor[:, :, :, :-1]  # remove bg
        # convert keypoint map to coordinate
        keypoint_param = keypoints_2d.keypoint_map_to_gaussian_coordinate(
            keypoint_map,
            use_hard_max_as_anchors=self.options["use_hard_max_as_anchors"] if
            "use_hard_max_as_anchors" in self.options else None
        )
        # Remark: keypoint_param has been scaled according to aspect ratio

        # keypoint_map_shape = tmf.get_shape(keypoint_map)
        # batch_size = keypoint_map_shape[0]
        batch_size = tmf.get_shape(keypoint_map)[0]

        keypoint_prob = tf.ones([batch_size, tmf.get_shape(keypoint_map)[3]], dtype=keypoint_map.dtype)

        keypoint_param = tf.concat([
            keypoint_param[:, :, :2],
            tf.reduce_mean(keypoint_param[:, :, 2:4], axis=2, keep_dims=True)
        ], axis=2)   # use isotropic gaussian

        return keypoint_param, keypoint_prob

    def heatmap2structure_basic(self, heatmap_tensor):
        keypoint_param, _ = self.heatmap2structure_internal(heatmap_tensor)
        keypoint_param = keypoint_param[:, :, :self.structure_param_num]
        return keypoint_param

    def heatmap2structure(self, heatmap_tensor):
        return self.heatmap2structure_internal(heatmap_tensor) + (heatmap_tensor, None)

    def heatmap2structure_poststep(self, structure_pack):

        # extra outputs
        extra_outputs = dict()
        extra_outputs["for_decoder"] = dict()

        # get necessary information
        keypoint_param, keypoint_prob, heatmap_tensor = structure_pack

        # compute keypoints
        keypoint_map = heatmap_tensor[:, :, :, :-1]  # remove bg
        # get image size
        actual_h, actual_w, full_h, full_w = self.image_size(
            tmf.get_shape(keypoint_map)[1], tmf.get_shape(keypoint_map)[2]
        )

        batch_size = tmf.get_shape(keypoint_param)[0]
        main_batch_size = batch_size // 2 if self.use_random_transform() else batch_size

        output_shape = tmf.get_shape(keypoint_map)
        out_h = output_shape[1]
        out_w = output_shape[2]
        out_ah = int(out_h * (actual_h/full_h))
        out_aw = int(out_w * (actual_w/full_w))
        out_scaling = math.sqrt((actual_h/full_h) * (actual_w/full_w))

        if "keypoint_concentration_loss_weight" in self.options and \
                rbool(self.options["keypoint_concentration_loss_weight"]):
            gaussian_spatial_entropy = tf.reduce_mean(
                tf.reduce_sum(
                    keypoints_2d.gaussian2d_exp_entropy(keypoint_param, stddev_scaling=out_scaling),
                    axis=1
                ), axis=0)
            keypoint_concentration_loss = gaussian_spatial_entropy * self.options["keypoint_concentration_loss_weight"]
            keypoint_concentration_loss.disp_name = 'concentration'
            tf.add_to_collection(
                "aux_loss", keypoint_concentration_loss)

        if "keypoint_separation_loss_weight" in self.options and \
                rbool(self.options["keypoint_separation_loss_weight"]):

            assert "keypoint_separation_bandwidth" in self.options, "keypoint_separation_bandwidth must be defined"
            keypoint_separation_bandwidth = self.options["keypoint_separation_bandwidth"] * out_scaling

            keypoint_loc = keypoint_param[:, :, :2]
            keypoint_dist = tf.reduce_sum(tf.square(
                tf.expand_dims(keypoint_loc, axis=1) - tf.expand_dims(keypoint_loc, axis=2)), axis=3)
            keypoint_vicinity = tf.exp(-keypoint_dist / (2*(keypoint_separation_bandwidth**2)))  # quadratic
            keypoint_vicinity = tf.where(
                tf.eye(tmf.get_shape(keypoint_loc)[1], batch_shape=[batch_size]) > 0,
                tf.zeros_like(keypoint_vicinity), keypoint_vicinity
            )
            keypoint_separation_loss = tf.reduce_sum(keypoint_vicinity) / batch_size
            keypoint_separation_loss *= self.options["keypoint_separation_loss_weight"]
            keypoint_separation_loss.disp_name = 'kp_separate'
            tgu.add_to_aux_loss(keypoint_separation_loss)

        regularized_map_full = keypoints_2d.gaussian_coordinate_to_keypoint_map(
            keypoint_param, tmf.get_shape(keypoint_map)[1], tmf.get_shape(keypoint_map)[2]
        )

        # heatmap for patch_features
        background_weights = tf.pad(
            tf.ones([batch_size, out_ah, out_aw, 1], dtype=regularized_map_full.dtype) / (actual_h * actual_w),
            [
                [0, 0], [(out_h - out_ah) // 2, (out_h - out_ah) - (out_h - out_ah) // 2],
                [(out_w - out_aw) // 2, (out_w - out_aw) - (out_w - out_aw) // 2], [0, 0]
            ], mode="CONSTANT", constant_values=0
        )

        keypoint_param_for_patch_features = keypoint_param
        heatmap_stddev_for_patch_features = None

        if heatmap_stddev_for_patch_features is not None:
            keypoint_param_for_patch_features = tf.concat([
                keypoint_param[:, :, :2], heatmap_stddev_for_patch_features
            ], axis=2)

        regularized_map_full = keypoints_2d.gaussian_coordinate_to_keypoint_map(
            keypoint_param_for_patch_features, tmf.get_shape(keypoint_map)[1], tmf.get_shape(keypoint_map)[2]
        )

        # visualize the computed gaussian
        regularized_map = regularized_map_full[:main_batch_size]

        cropped_regularized_map = \
            regularized_map[:, (full_h-actual_h)//2:(full_h+actual_h)//2, (full_w-actual_w)//2:(full_w+actual_w)//2]

        extra_outputs["save"] = dict(
            regularized_map=cropped_regularized_map,
            keypoint_prob=keypoint_prob[:main_batch_size]
        )

        keypoint_param = keypoint_param[:, :, :self.structure_param_num]

        structure_param = keypoint_param

        return structure_param, extra_outputs

    def cleanup_augmentation_patchfeatures(self, patch_features, aug_cache):
        main_batch_size = aug_cache["main_batch_size"]
        batch_size = tmf.get_shape(patch_features)[0]
        if batch_size == main_batch_size:
            return patch_features
        return patch_features[:main_batch_size]

    def cleanup_augmentation_structure(self, structure_param, aug_cache, condition_tensor=None):

        actual_h, actual_w, full_h, full_w = self.image_size(
            tmf.get_shape(aug_cache["original_image"])[0],
            tmf.get_shape(aug_cache["original_image"])[1]
        )
        full_a = full_w / full_h
        af_scaling = math.sqrt((actual_h / full_h)*(actual_w / full_w))

        if not self.use_random_transform():

            keypoint_param = structure_param
            batch_size = tmf.get_shape(structure_param)[0]

        else:

            with tf.variable_scope("transform_invariance"):

                lm_tps_cp = aug_cache["lm_tps"]["transform"]
                lm_tps_fp = aug_cache["lm_tps"]["control_points"]
                tps_transform = aug_cache["tps"]["transform"]
                tps_n_points = aug_cache["tps"]["num_points"]
                use_lm_tps = aug_cache["use_lm_tps"]
                transform = aug_cache["sim_transform"]

                batch_size = tmf.get_shape(structure_param)[0] // 2
                # keypoint_num = tmf.get_shape(structure_param)[1]

                # transform keypoints and match keypoints
                keypoint_param2 = structure_param[batch_size:, :, :2]
                keypoint_param = structure_param[:batch_size, :, :2]

                # keypoint matching
                kp1 = self.coordinate_to_stn(keypoint_param, aspect_ratio=full_a)
                kp2 = self.coordinate_to_stn(keypoint_param2, aspect_ratio=full_a)
                kp1h_from2 = (
                    pt.wrap(kp2).
                    coordinate_inv_transformer(transform)
                )
                kp1from2 = tf.where(
                    tf.tile(tmf.expand_dims(use_lm_tps, axis=-1, ndims=2), [1]+tmf.get_shape(kp2)[1:]),
                    kp1h_from2.coordinate_inv_transformer_tps(None, None, lm_tps_cp, fp_more=lm_tps_fp),
                    kp1h_from2.coordinate_inv_transformer_tps(tps_n_points, tps_n_points, tps_transform)
                )
                kp_diff_loss = tf.reduce_sum(
                    tf.reduce_sum(tf.square(kp1from2 - kp1), axis=[0, 1]) *
                    np.array([full_a, 1/full_a])) / (af_scaling * batch_size)
                # remark: x,y: [-1,1]x[-1,1] --> [-aspect,+aspect]x[-1/aspect,+1/aspect], note the square
                transform_invariant_loss = self.options["keypoint_transform_loss_weight"] * kp_diff_loss
                tgu.add_to_aux_loss(transform_invariant_loss, "enc_transform")

        # optical flow
        of_condition = None
        if condition_tensor is not None:
            assert condition_tensor is not None, "need optical flow condition"
            for v in condition_tensor:
                if v["type"] == "optical_flow":
                    of_condition = v

        optical_flow_transform_loss_weight = None
        if "optical_flow_transform_loss_weight" in self.options:
            optical_flow_transform_loss_weight = self.options["optical_flow_transform_loss_weight"]

        if optical_flow_transform_loss_weight is None:
            if of_condition is not None and "keypoint_transform_loss_weight" in self.options:
                optical_flow_transform_loss_weight = self.options["keypoint_transform_loss_weight"]

        optical_flow_strength_loss_weight = None
        if "optical_flow_strength_loss_weight" in self.options:
            optical_flow_strength_loss_weight = self.options["optical_flow_strength_loss_weight"]

        if ptu.default_phase() == pt.Phase.train and \
                (rbool(optical_flow_transform_loss_weight) or rbool(optical_flow_strength_loss_weight)):

            assert of_condition is not None, "need optical flow condition"

            # coordinate before padding
            pre_keypoint_param = keypoint_param[:, :, :2]
            scaling_factor = np.array(self.target_input_size) / np.array(self.input_size)
            pre_keypoint_param = keypoints_2d.scale_keypoint_param(
                pre_keypoint_param, scaling_factor, src_aspect_ratio=full_a)

            # only use valid
            ind_offset = tf.reshape(of_condition["offset"], [-1])
            flow_map = of_condition["flow"]  # [batch_size, h, w, 2]
            valid_mask = tf.not_equal(ind_offset, 0)

            # interpolation mask
            flow_h, flow_w = tmf.get_shape(flow_map)[1:3]

            if rbool(optical_flow_transform_loss_weight):

                pre_interp_weights = keypoints_2d.gaussian_coordinate_to_keypoint_map(tf.concat([
                    pre_keypoint_param,
                    tf.ones_like(pre_keypoint_param[:, :, -1:]) / math.sqrt(flow_h * flow_w)
                ], axis=2), km_h=flow_h, km_w=flow_w)  # [batch_size, h, w, keypoint_num]
                pre_interp_weights /= tf.reduce_sum(pre_interp_weights, axis=[1, 2], keep_dims=True) + tmf.epsilon

                # pointwise flow
                next_ind = np.arange(batch_size) + ind_offset
                next_keypoint_param = tf.gather(pre_keypoint_param, next_ind)
                pointwise_flow = tf.reduce_sum(
                    tf.expand_dims(flow_map, axis=3)*tf.expand_dims(pre_interp_weights, axis=4),
                    axis=[1, 2]
                )

                # flow transform constraint
                next_keypoint_param_2 = pre_keypoint_param + pointwise_flow
                kp_of_trans_loss = tf.reduce_mean(tf.boolean_mask(
                    tmf.sum_per_sample(tf.square(next_keypoint_param_2 - next_keypoint_param)),
                    mask=valid_mask
                ))
                optical_flow_transform_loss = kp_of_trans_loss * optical_flow_transform_loss_weight
                tgu.add_to_aux_loss(optical_flow_transform_loss, "flow_trans")

            if rbool(optical_flow_strength_loss_weight):

                pre_interp_weights = keypoints_2d.gaussian_coordinate_to_keypoint_map(tf.concat([
                    pre_keypoint_param,
                    tf.ones_like(pre_keypoint_param[:, :, -1:]) * (1/16)  #self.base_gaussian_stddev
                ], axis=2), km_h=flow_h, km_w=flow_w)  # [batch_size, h, w, keypoint_num]
                pre_interp_weights /= tf.reduce_sum(pre_interp_weights, axis=[1, 2], keep_dims=True) + tmf.epsilon

                kp_of_strength_loss = tf.reduce_mean(tmf.sum_per_sample(
                    tf.boolean_mask(pre_interp_weights, mask=valid_mask) *
                    tf.sqrt(tf.reduce_sum(
                        tf.square(tf.boolean_mask(flow_map, mask=valid_mask)), axis=3, keep_dims=True))
                ))
                # kp_of_strength_loss = 1/(kp_of_strength_loss+1)
                kp_of_strength_loss = -kp_of_strength_loss
                optical_flow_strength_loss = kp_of_strength_loss * optical_flow_strength_loss_weight
                tgu.add_to_aux_loss(optical_flow_strength_loss, "flow_strength")

        # scale the parameters based on the padding ------
        if self.target_input_size is not None:
            assert self.input_size is not None, "self.input_size must be specified if self.target_input_size"
            scaling_factor = np.array(self.target_input_size) / np.array(self.input_size)
            keypoint_param = keypoints_2d.scale_keypoint_param(
                keypoint_param, scaling_factor,
                src_aspect_ratio=full_a)

        return keypoint_param

    def structure_param2euclidean(self, structure_param):
        return keypoints_2d.gaussian2dparam_to_recon_code(structure_param)

    def coordinate_to_stn(self, keypoint_param, aspect_ratio):

        # Remark: keypoint_param is scaled according to aspect_ratio
        # need to make it [0, 1] x [0, 1]
        return tf.concat([  # swap yx to xy, and [0,1] -> [-1,1]
            keypoint_param[:, :, 1:2] / math.sqrt(aspect_ratio) * 2 - 1.,
            keypoint_param[:, :, 0:1] * math.sqrt(aspect_ratio) * 2 - 1.
        ], axis=2)


Factory = BasicFactory
