import zutils.tf_math_funcs as tmf
import tensorflow as tf
import math
import numpy as np
import collections

epsilon = tmf.epsilon


param_num_2d_gaussian = 5
gaussian_2d_base_stddev = 1/8


def yx_grid_map(h, w, dtype, aspect_ratio=1.):
    arsr = math.sqrt(aspect_ratio)

    h_max = 1 / arsr
    w_max = arsr

    y_map = tmf.normalized_index_template(h, dtype=dtype, offset=0.5) * h_max
    y_map = tmf.expand_dims(tmf.expand_dims(y_map, axis=0, ndims=1), axis=-1, ndims=3)
    x_map = tmf.normalized_index_template(w, dtype=dtype, offset=0.5) * w_max
    x_map = tmf.expand_dims(tmf.expand_dims(x_map, axis=0, ndims=2), axis=-1, ndims=2)
    yx_map = tf.concat(
        [tf.tile(y_map, [1, 1, w, 1, 1]), tf.tile(x_map, [1, h, 1, 1, 1])],
        axis=4
    )  # [1, H, W, 1, 2]
    return yx_map


def keypoint_map_to_gaussian_coordinate(keypoint_map, diag_cov=None, use_hard_max_as_anchors=None):
    """

    :param keypoint_map:
    :return:
    """

    if diag_cov is None:
        diag_cov = False
    if use_hard_max_as_anchors is None:
        use_hard_max_as_anchors = False

    keypoint_shape = tmf.get_shape(keypoint_map)
    batch_size = keypoint_shape[0]
    km_h = keypoint_shape[1]
    km_w = keypoint_shape[2]
    keypoint_num = keypoint_shape[3]
    km_a = km_w / km_h
    nh = 1/math.sqrt(km_a)
    nw = 1*math.sqrt(km_a)

    def cyclic_clip(a):
        c = tmf.expand_dims([nh, nw], axis=0, ndims=len(tmf.get_shape(a))-1)
        return tf.where(a > c, a - c, tf.where(a < 0, a + c, a))

    yx_map_raw = yx_grid_map(km_h, km_w, keypoint_map.dtype, aspect_ratio=km_a)  # [1, H, W, 1, 2]
    yx_map_raw = tf.reshape(yx_map_raw, [1, km_h * km_w, 1, 2])  # [1, H*W, 1, 2]

    k_map = tf.reshape(keypoint_map, [batch_size, km_h * km_w, keypoint_num, 1])  # [batch_size, H*W, keypoint_num, 1]
    k_summed = tf.reduce_sum(k_map, axis=1) + epsilon  # [batch_size, H*W, keypoint_num, 1]

    if use_hard_max_as_anchors:
        # figure out the hard argmax
        hard_flatten_idx = tf.argmax(tf.squeeze(k_map, axis=3), 1, output_type=tf.int64)  # [batch_size, keypoint_num]
        yx_map_flatten_single = tf.reshape(yx_map_raw, [km_h*km_w, 2])  # [H*W, 2]
        hard_yx_coordinates = tf.gather(yx_map_flatten_single, hard_flatten_idx)  # [batch_size, keypoint_num, 2]
        # anchor for computing the mean
        anchor_map = tf.expand_dims(hard_yx_coordinates, axis=1)
        # anchor_map: [batch_size, 1, keypoint_num, 2]
        hw_map = tmf.expand_dims(tf.cast([nh, nw], k_map.dtype), axis=0, ndims=3)  # [batch_size, 1, keypoint_num, 2]
        offset_map = anchor_map - hw_map * 0.5  # [batch_size, 1, keypoint_num, 2]
        yx_map = yx_map_raw - offset_map   # [batch_size, H*W, keypoint_num, 2]
        yx_map = cyclic_clip(yx_map)
        # yx_map: [batch_size, H*W, keypoint_num, 2]
    else:
        yx_map = yx_map_raw

    # weighted mean
    yx_mean = tf.reduce_sum(k_map * yx_map, axis=1) / k_summed   # [batch_size, keypoint_num, 2]

    # weighted covariance
    yx_offsets = yx_map - tf.expand_dims(yx_mean, axis=1)
    yx_elt_selfcov = tf.square(yx_offsets)
    yx_elt_crosscov = yx_offsets[:, :, :, 0:1] * yx_offsets[:, :, :, 1:2]
    yx_elt_sccov = tf.concat([yx_elt_selfcov, yx_elt_crosscov], axis=3)
    yx_sccov = tf.reduce_sum(yx_elt_sccov * k_map, axis=1) / k_summed   # [batch_size, keypoint_num, 3]
    yx_self_stddev = tf.sqrt(yx_sccov[:, :, 0:2])
    yx_corr = tf.expand_dims(
        yx_sccov[:, :, 2] / ((yx_self_stddev[:, :, 0] * yx_self_stddev[:, :, 1])+epsilon),
        axis=-1
    )

    if diag_cov:
        yx_corr = tf.zeros_like(yx_corr)

    if use_hard_max_as_anchors:
        yx_mean = cyclic_clip(yx_mean + tf.squeeze(offset_map, axis=1))

    yx_mean_stddev_corr = tf.concat(
        [yx_mean, yx_self_stddev, yx_corr], axis=2
    )   # [batch_size, keypoint_num, 5]: y_mean, x_mean, y_stddev, x_stddev, yx_corr

    return yx_mean_stddev_corr


def gaussian_coordinate_to_keypoint_map(yx_mean_stddev_corr, km_h, km_w, dtype=None):

    input_shape = tmf.get_shape(yx_mean_stddev_corr)
    assert len(input_shape) == 3, "wrong rank"

    input_tensor_list = list()
    input_tensor_list.append(yx_mean_stddev_corr)
    if input_shape[2] < 3:
        input_tensor_list.append(
            tf.ones(input_shape[:2] + [2], dtype=yx_mean_stddev_corr.dtype) * gaussian_2d_base_stddev
        )
    elif input_shape[2] < 4:
        input_tensor_list.append(
            yx_mean_stddev_corr[:, :, 2:3]
        )
    if input_shape[2] < 5:
        input_tensor_list.append(
            tf.zeros(input_shape[:2] + [1], dtype=yx_mean_stddev_corr.dtype)
        )
    yx_mean_stddev_corr = tf.concat(
        input_tensor_list, axis=2
    )

    input_shape = tmf.get_shape(yx_mean_stddev_corr)
    assert input_shape[2] == 5, "wrong parameter number"

    if dtype is None:
        dtype = yx_mean_stddev_corr.dtype

    # batch_size = input_shape[0]
    # keypoint_num = input_shape[1]

    yx_map = yx_grid_map(km_h, km_w, dtype, aspect_ratio=km_w/km_h)       # [1, H, W, 1, 2]

    p_map = tmf.expand_dims(yx_mean_stddev_corr, axis=1, ndims=2)    # [batch_size, 1, 1, keypoint_num, 5]

    det_map = tmf.expand_dims(
        gaussian2d_det(yx_mean_stddev_corr),
        axis=1, ndims=2
    )

    yx_zm_map = yx_map - p_map[:, :, :, :, 0:2]   # y, x : zero mean
    yx_zm_map_2 = tf.square(yx_zm_map)
    m_map = p_map[:, :, :, :, 2:]  # sigma_y, sigma_x, corr_yx
    m_map_2 = tf.square(m_map)

    u_numerator = (
        yx_zm_map_2[:, :, :, :, 0] * m_map_2[:, :, :, :, 1] +
        yx_zm_map_2[:, :, :, :, 1] * m_map_2[:, :, :, :, 0] -
        2. * tf.reduce_prod(yx_zm_map, axis=4) * tf.reduce_prod(m_map, axis=4))
    u_denominator = (tf.square(m_map_2[:, :, :, :, 2])-1.) * m_map_2[:, :, :, :, 0] * m_map_2[:, :, :, :, 1] - epsilon
    keypoint_map = tmf.safe_exp(0.5*(u_numerator/u_denominator)) / ((2.*math.pi*det_map+epsilon)*(km_h*km_w))
    keypoint_map /= km_h * km_w     # normalize to probability mass

    return keypoint_map


def gaussian2dparam_to_real(p):
    assert len(tmf.get_shape(p)) == 3, "wrong rank"
    param_num = tmf.get_shape(p)[-1]
    assert 2 <= param_num <= 5, "wrong param number"

    tensor_list = list()
    tensor_list.append(tmf.inv_sigmoid(p[:, :, 0:2]))
    if param_num >= 4:
        tensor_list.append(tmf.inv_atanh_sigmoid(p[:, :, 2:4] / gaussian_2d_base_stddev))
        if param_num == 5:
            tensor_list.append(tmf.atanh(p[:, :, 4:5]))
    else:
        tensor_list.append(tmf.inv_atanh_sigmoid(p[:, :, 2:3] / gaussian_2d_base_stddev))

    return tf.concat(tensor_list, axis=2)


def real_to_gaussian2dparam(r):
    assert len(tmf.get_shape(r)) == 3, "wrong rank"
    param_num = tmf.get_shape(r)[-1]
    assert 2 <= param_num <= 5, "wrong param number"

    tensor_list = list()
    tensor_list.append(tf.nn.sigmoid(r[:, :, 0:2]))
    if param_num >= 4:
        tensor_list.append(tmf.atanh_sigmoid(r[:, :, 2:4]) * gaussian_2d_base_stddev)
        if param_num == 5:
            tensor_list.append(tf.nn.tanh(r[:, :, 4:5]))
    else:
        tensor_list.append(tmf.atanh_sigmoid(r[:, :, 2:3]) * gaussian_2d_base_stddev)

    return tf.concat(tensor_list, axis=2)


def gaussian2dparam_to_recon_code(p):
    assert len(tmf.get_shape(p)) == 3, "wrong rank"
    param_num = tmf.get_shape(p)[-1]
    assert 2 <= param_num <= 5, "wrong param number"

    tensor_list = list()
    tensor_list.append(p[:, :, :2])
    if param_num >= 4:
        tensor_list.append(tf.log(p[:, :, 2:4] / gaussian_2d_base_stddev + epsilon))
        if param_num == 5:
            tensor_list.append(tmf.atanh(p[:, :, 4:5]))
    else:
        tensor_list.append(tf.log(p[:, :, 2:3] / gaussian_2d_base_stddev + epsilon))

    return tf.concat(tensor_list, axis=2)


def real_to_recon_code(r):
    p = real_to_gaussian2dparam(r)

    param_num = tmf.get_shape(p)[-1]

    tensor_list = list()
    tensor_list.append(p[:, :, :2])
    if param_num >= 4:
        tensor_list.append(tf.log(p[:, :, 2:4] / gaussian_2d_base_stddev + epsilon))
        if param_num == 5:
            tensor_list.append(r[:, :, 4:5])
    else:
        tensor_list.append(tf.log(p[:, :, 2:3] / gaussian_2d_base_stddev + epsilon))

    return tf.concat(tensor_list, axis=2)


def gaussian2d_det(p, no_mean_input=False):
    assert len(tmf.get_shape(p)) == 3, "wrong rank"
    if not no_mean_input:
        p = p[:, :, 2:]
    p_shape = tmf.get_shape(p)
    if p_shape[-1] == 0:
        d = tf.ones(p_shape[:-1]+[1], dtype=p.dtype) * math.pow(gaussian_2d_base_stddev, 4.)
    elif p_shape[-1] == 1:
        d = tf.pow(p[:, :, 0], 4.)
    elif p_shape[-1] == 2:
        d = tf.square(p[:, :, 0] * p[:, :, 1])
    elif p_shape[-1] == 3:
        d = (1. - tf.square(p[:, :, 2])) * tf.square(p[:, :, 0] * p[:, :, 1])
    else:
        raise ValueError("too many parameters")
    return d


def gaussian2d_entropy(p, no_mean_input=False):
    assert len(tmf.get_shape(p)) == 3, "wrong rank"
    if not no_mean_input:
        p = p[:, :, 2:]
    p_shape = tmf.get_shape(p)
    z = (math.log(2*math.pi) + 1.)
    if p_shape[-1] == 0:
        d = tf.ones(p_shape[:-1]+[1], dtype=p.dtype) * \
            (z + 2. * math.log(gaussian_2d_base_stddev))
    elif p_shape[-1] == 1:
        d = z + 2. * tf.log(p[:, :, 0] + epsilon)
    elif p_shape[-1] == 2:
        d = z + (
            tf.log(p[:, :, 0]+epsilon) +
            tf.log(p[:, :, 1]+epsilon)
        )
    elif p_shape[-1] == 3:
        d = z + (
            0.5 * tf.log(1.-tf.square(p[:, :, 2])+epsilon) +
            tf.log(p[:, :, 0]+epsilon) +
            tf.log(p[:, :, 1]+epsilon)
        )
    else:
        raise ValueError("too many parameters")
    return d


def gaussian2d_exp_entropy(p, no_mean_input=False, stddev_scaling=1.):
    assert len(tmf.get_shape(p)) == 3, "wrong rank"
    if not no_mean_input:
        p = p[:, :, 2:]
    p_shape = tmf.get_shape(p)
    z = math.exp(math.log(2*math.pi) + 1.)
    if p_shape[-1] == 0:
        d = tf.ones(p_shape[:-1]+[1], dtype=p.dtype) * \
            (z * math.exp(2. * math.log(gaussian_2d_base_stddev)))
    elif p_shape[-1] == 1:
        d = z * tf.square(p[:, :, 0])
    elif p_shape[-1] == 2:
        d = z * p[:, :, 0] * p[:, :, 1]
    elif p_shape[-1] == 3:
        d = z * p[:, :, 0] * p[:, :, 1] * tf.sqrt(1.-tf.square(p[:, :, 2]))
    else:
        raise ValueError("too many parameters")
    return d / (stddev_scaling**2)


def gaussian2d_axis_balancing(p, no_mean_input=False):
    assert len(tmf.get_shape(p)) == 3, "wrong rank"
    if not no_mean_input:
        p = p[:, :, 2:]

    p_shape = tmf.get_shape(p)
    if p_shape[-1] == 0:
        # b = tf.ones(p_shape[:-1]+[1], dtype=p.dtype) * math.log(epsilon)
        b = tf.constant(0, dtype=p.dtype)
    else:
        p2 = tf.square(p)
        if p_shape[-1] == 1:
            x = p2[:, :, 0]
            y = x
            q = 0.
        elif p_shape[-1] == 2:
            x = p2[:, :, 0]
            y = p2[:, :, 1]
            q = 0.
        elif p_shape[-1] == 3:
            x = p2[:, :, 0]
            y = p2[:, :, 1]
            q = p2[:, :, 2]
        else:
            raise ValueError("too many parameters")
        x_plus_y = x+y
        # b = tf.log(tf.square(x_plus_y) + 4.*(q-1.)*x*y + epsilon) - 2.*tf.log(x_plus_y + epsilon)
        b = 1. + (4. * (q - 1.) * x * y) / tf.square(x_plus_y)
    return b


"""
def gaussian2d_kl(p, q):
    assert len(tmf.get_shape(p)) == 3, "wrong rank"
    assert tmf.get_shape(p)[-1] == 5, "wrong param number"
    assert len(tmf.get_shape(q)) == 3, "wrong rank"
    assert tmf.get_shape(q)[-1] == 5, "wrong param number"
"""


def keypoint_map_boundary_mask(h, w, padding_h, padding_w, dtype=np.float32):

    km_mask = np.ones([h, w], dtype=dtype)
    km_mask[:padding_h, :] = 0.
    km_mask[-padding_h:, :] = 0.
    km_mask[:, :padding_w] = 0.
    km_mask[:, -padding_w:] = 0.

    km_mask = np.reshape(km_mask, [1, h, w, 1])

    return km_mask


def keypoint_map_boundary_mask_with_bg(h, w, padding_h, padding_w, keypoint_num, dtype=np.float32):

    km_mask = keypoint_map_boundary_mask(h, w, padding_h, padding_w, dtype=np.float32)
    full_mask = np.tile(km_mask, [1, 1, 1, keypoint_num+1])
    full_mask[:, :, :, -1] = 1.

    return full_mask


def keypoint_map_depth_normalization_with_fake_bg(keypoint_map):
    bg_prob = 1. / (tmf.get_shape(keypoint_map)[1] * tmf.get_shape(keypoint_map)[2])
    keypoint_map_z = tf.reduce_sum(keypoint_map, axis=3, keep_dims=True) + bg_prob
    keypoint_map /= keypoint_map_z
    normalized_bg_prob = bg_prob / tf.squeeze(keypoint_map_z, axis=3)
    return keypoint_map, normalized_bg_prob


def keypoint_map_depth_entropy_with_fake_bg(keypoint_map):
    keypoint_map, normalized_bg_prob = keypoint_map_depth_normalization_with_fake_bg(keypoint_map)
    neg_heatmap_entropy = \
        tf.reduce_sum(keypoint_map * tf.log(keypoint_map + epsilon), axis=3) + \
        normalized_bg_prob * tf.log(normalized_bg_prob + epsilon)
    total_heatmap_entropy = -tf.reduce_mean(neg_heatmap_entropy)
    return total_heatmap_entropy


def keypoint_map_depth_normalization_with_real_bg(keypoint_map):
    keypoint_map_z = tf.reduce_sum(keypoint_map, axis=3, keep_dims=True) + epsilon
    keypoint_map /= keypoint_map_z
    return keypoint_map


def keypoint_map_depth_entropy_with_real_bg(keypoint_map):
    keypoint_map = keypoint_map_depth_normalization_with_real_bg(keypoint_map)
    neg_heatmap_entropy = \
        tf.reduce_sum(keypoint_map * tf.log(keypoint_map + epsilon), axis=3)
    total_heatmap_entropy = -tf.reduce_mean(neg_heatmap_entropy)
    return total_heatmap_entropy


def scale_keypoint_param(keypoint_param, scaling_factor, src_aspect_ratio):

    # scaling_factor is respect to point coordinate, not the image size

    assert len(tmf.get_shape(keypoint_param)) == 3, "wrong dimension for keypoint_param"
    num_params = tmf.get_shape(keypoint_param)[2]

    src_arsr = math.sqrt(src_aspect_ratio)

    if isinstance(scaling_factor, np.ndarray):
        scaling_factor = scaling_factor.flatten()
        scaling_factor = scaling_factor.tolist()

    scaling_factor = np.array(scaling_factor)

    if hasattr(scaling_factor, "shape") and not scaling_factor.shape:
        scaling_factor = scaling_factor.tolist()
        dst_arsr = src_arsr
    if not hasattr(scaling_factor, "__len__"):
        scaling_factor = scaling_factor
        dst_arsr = src_arsr
    elif len(scaling_factor) == 1:
        scaling_factor = scaling_factor[0]
        dst_arsr = src_arsr
    elif len(scaling_factor) == 2:      # y, x
        dst_arsr = (
            src_arsr
            / scaling_factor[1]     # x scaling, equals to shrink width by scaling_factor[1]
            * scaling_factor[0]     # y scaling
        )
    else:
        raise ValueError("scaling_factor should be a scalar or 2-D vector")

    src_center_pt = tf.reshape(tf.constant([0.5/src_arsr, 0.5*src_arsr]), [1, 1, 2])
    dst_center_pt = tf.reshape(tf.constant([0.5/dst_arsr, 0.5*dst_arsr]), [1, 1, 2])

    if hasattr(scaling_factor, "__len__") and len(scaling_factor) == 2:

        scaling_factor = tf.constant(scaling_factor, dtype=keypoint_param.dtype)
        scaling_factor = tf.reshape(scaling_factor, [1, 1, 2])
        q_yx = (keypoint_param[:, :, 0:2] - src_center_pt) * scaling_factor + dst_center_pt
        if num_params < 3:
            return q_yx
        if num_params == 3:
            return tf.concat([q_yx, keypoint_param[:, :, 2:3] * scaling_factor], axis=2)
        elif num_params == 4:
            return tf.concat([q_yx, keypoint_param[:, :, 2:4] * scaling_factor], axis=2)
        elif num_params == 5:
            return tf.concat([q_yx, keypoint_param[:, :, 2:4] * scaling_factor, keypoint_param[:, :, 4:5]], axis=2)

    else:

        q_yx = (keypoint_param[:, :, 0:2] - src_center_pt) * scaling_factor + dst_center_pt
        if num_params < 3:
            return q_yx
        if num_params == 3:
            return tf.concat([q_yx, keypoint_param[:, :, 2:3] * scaling_factor], axis=2)
        elif num_params == 4:
            return tf.concat([q_yx, keypoint_param[:, :, 2:4] * scaling_factor], axis=2)
        elif num_params == 5:
            return tf.concat([q_yx, keypoint_param[:, :, 2:4] * scaling_factor, keypoint_param[:, :, 4:]], axis=2)


def get_map_aspect_ratio(a):

    s = tmf.get_shape(a)
    return s[2] / s[1]


def parse_landmark_condition_tensor_single(condition_tensor, img_size=None, full_img_size=None):

    keypoint_struct = list(filter(lambda x: x["type"] == "landmark", condition_tensor))
    assert len(keypoint_struct) == 1, "need exact one condition for keypoints"
    keypoint_struct = keypoint_struct[0]
    return parse_landmark_condition_struct(keypoint_struct, img_size, full_img_size)


def parse_landmark_condition_struct(keypoint_struct, img_size=None, full_img_size=None):

    # parse the landmark locations
    keypoint_param = keypoint_struct["location"]  # [0,1/sqrt(a)]*[0,1/sqrt(a)]
    keypoint_gate = keypoint_struct["gate"]

    # canonicalize the location representation
    if img_size is not None and img_size != full_img_size:       # y,x
        y_factor = img_size[0]/full_img_size[0]
        x_factor = img_size[1]/full_img_size[1]
        keypoint_param = (keypoint_param-0.5) * tmf.expand_dims([y_factor, x_factor], axis=0, ndims=2) + 0.5
        a = full_img_size[1] / full_img_size[0]
    """
    if a != 1:
        arsr = math.sqrt(a)
        keypoint_param *= tmf.expand_dims([1/arsr, arsr], axis=0, ndims=2)
    """

    return keypoint_param, keypoint_gate

