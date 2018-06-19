import tensorflow as tf
import numpy as np
import math
from copy import copy
from collections import Iterable
from functools import reduce

epsilon = 1e-6
epsilon2 = 1e-160


def is_tf_data(a):
    return isinstance(a, (tf.Tensor, tf.Variable))


def get_shape(a):

    def int_r(v):
        if v is None:
            return 0
        else:
            return int(v)

    if is_tf_data(a):
        s = list(int_r(v) for v in a.get_shape())
    elif isinstance(a, np.ndarray):
        s = a.shape
    elif hasattr(a, "shape"):
        s = list(int_r(v) for v in a.shape)
    else:
        raise ValueError("get_shape supports tf.Tensor/tf.Variable and np.ndarray")
    return s


def shape(a):
    if is_tf_data(a):
        s = tf.shape(a.get_shape)
    elif isinstance(a, np.ndarray):
        s = a.shape
    else:
        raise ValueError("get_shape supports tf.Tensor/tf.Variable and np.ndarray")
    return s


def expand_minimum_ndim(a, target_dim, axis=-1):
    if is_tf_data(a):
        cur_dim = len(a.get_shape())
        b = a
        for i in range(cur_dim, target_dim):
            b = tf.expand_dims(b, axis=axis)
    else:
        if isinstance(a, np.ndarray):
            b = a
        else:
            b = np.array(a).astype(np.float32)
        cur_dim = len(b.shape)
        for i in range(cur_dim, target_dim):
            b = np.expand_dims(b, axis=axis)
    return b


def atanh(z):
    return 0.5 * (tf.log(1 + z) - tf.log(1 - z + epsilon))


def inv_sigmoid(x):
    return tf.log(x+epsilon) + tf.log(1-x+epsilon)


def safe_exp(z, ratio_allowed=0.5):
    if z.dtype == tf.float32:
        max_float = np.finfo('f').max
    elif z.dtype == tf.float64:
        max_float = np.finfo('d').max
    else:
        raise ValueError("safe_exp only works with float32 and float64")
    max_exp_allowed = math.log(max_float) * ratio_allowed
    return tf.exp(tf.maximum(tf.minimum(z, max_exp_allowed), -max_exp_allowed))


def atanh_sigmoid(z):
    # map R to R+
    return 0.5 * tf.log(1.0 + 2.0 * safe_exp(z, 0.49))


def inv_atanh_sigmoid(z):
    # map R+ to R
    return tf.log(0.5*(safe_exp(2*z, 0.49)-1.0)+epsilon)


def sqrt_exp(z):
    return tf.sqrt(tf.exp(z))


def log_clip(z, clip_point, top_point=None):
    if top_point is None:
        a = z
    else:
        a = tf.clip_by_value(z, 0.0, top_point)
    c = a / clip_point
    t = tf.where(c < 1, a, tf.log(c) * clip_point)
    return t


def sum_per_sample(a):
    ndim = len(a.get_shape())
    if ndim > 1:
        s = tf.reduce_sum(a, list(range(1, ndim)))
    else:
        s = a
    return s


def rep_sample(a, n):

    if not is_tf_data(n):
        if n == 1:
            return a

    s = get_shape(a)
    if np.any(np.array(list(j is None for j in s))):
        s = tf.shape(a)
    ndims = len(a.get_shape())

    b = tf.expand_dims(a, 1)
    if is_tf_data(n):
        n = tf.reshape(n, [1])
        r = [[1], n]
    else:
        r = [[1, n]]
    if ndims > 1:
        r += [[1]*(ndims-1)]
    r = tf.concat(values=r, axis=0)
    b = tf.tile(b, r)

    if is_tf_data(s):
        t = tf.concat(values=[s[0:1]*n, s[1:]], axis=0)
    else:
        t = [s[0] * n] + s[1:]
    c = tf.reshape(b, t)

    return c


def reduce_sample_block(a, n, reduce_func):

    if not is_tf_data(n):
        if n == 1:
            return a

    s = tf.shape(a)
    ndims = len(a.get_shape())

    batch_size = s[0:1]//n
    if is_tf_data(n):
        n = tf.reshape(n,[1])
        r = [batch_size, n]
    else:
        r = [batch_size, [n]]
    if ndims > 1:
        r += [s[1:]]
    r = tf.concat(values=r, axis=0)
    b = tf.reshape(a, r)

    c = reduce_func(b, axis=1)

    return c


def expand_dims(a, axis=None, ndims=1):
    b = a
    for i in range(ndims):
        b = tf.expand_dims(b, axis)
    return b


def concat_with_expand_dims(values, axis):
    concat_list = []
    is_tf = any([is_tf_data(v) for v in values])
    for v in values:
        if is_tf:
            if not is_tf_data(v):
                v = tf.constant(v)
        else:   # is np
            if not isinstance(v, np.ndarray):
                v = np.array(v)

        if is_tf:
            v = tf.expand_dims(v, axis=axis)
        else:
            v = np.expand_dims(v, axis=axis)
        concat_list.append(v)
    if is_tf:
        c = tf.concat(concat_list, axis=axis)
    else:
        c = np.concatenate(concat_list, axis=axis)
    return c


def interleave(values, axis, elt_mul=None):

    if elt_mul is None:
        all_n = list(get_shape(v)[axis] for v in values)
        min_n = min(all_n)
        elt_mul = list(n//min_n for n in all_n)

    if not isinstance(elt_mul, Iterable):
        elt_mul = [elt_mul] * len(values)

    single_shape = None
    is_np = True
    for v, m in zip(values, elt_mul):
        if single_shape is None:
            single_shape = get_shape(v)
            single_shape[axis] //= m
        else:
            assert single_shape[:axis] + single_shape[axis+1:] == \
                get_shape(v)[:axis] + get_shape(v)[axis+1:], \
                "all tensors should have the same shape, except axis"
            assert single_shape[axis] == get_shape(v)[axis]//m, \
                "inconsistent with elt mult"
        if not isinstance(v, np.ndarray):
            is_np = False

    ndim = len(single_shape)
    if axis < 0:
        axis = ndim+axis

    if is_np:
        the_expand_dims = np.expand_dims
        the_concat = np.concatenate
        the_reshape = np.reshape
    else:
        the_expand_dims = tf.expand_dims
        the_concat = tf.concat
        the_reshape = tf.reshape

    values_e = list()
    for v, m in zip(values, elt_mul):
        if m == 1:
            u = the_expand_dims(v, axis=axis-ndim)
        else:
            the_shape = single_shape[:(axis+1)] + [m] + single_shape[axis+1:]
            u = the_reshape(v, the_shape)
        values_e.append(u)

    concated_shape = copy(single_shape)
    concated_shape[axis] *= reduce(lambda x, y: x+y, elt_mul)
    c = the_concat(values_e, axis=axis+1)
    c = the_reshape(c, concated_shape)

    return c


def leaky_relu(x, alpha=0.2):
    return tf.where(tf.greater_equal(x, 0.), x, tf.multiply(alpha, x))


def elu(x):
    return tf.where(tf.greater_equal(x, 0.), x, tf.exp(x)-1.)


def average_pool_v1(input, ksize, strides, padding, name=None):

    assert strides[0] == 1 and strides[3] == 1, "unsupported strides"
    assert ksize[0] == 1 and ksize[3] == 1, "unsupported ksize"

    input = tf.expand_dims(input, axis=-1)
    strides += [1]
    ksize = [ksize[1], ksize[2], 1, 1, 1]
    kernel_elt_count = np.prod(np.array(ksize))
    kfilter = tf.ones(ksize, dtype=input.dtype) / kernel_elt_count
    output = tf.nn.conv3d(
        input, kfilter, strides=strides, padding=padding, name=name + "/pool" if name is not None else None)
    output = tf.squeeze(
        output, axis=-1, name=name + "/squeeze" if name is not None else None)

    return output


def average_pool_v2(input, ksize, strides, padding, name=None):

    assert strides[0] == 1 and strides[3] == 1, "unsupported strides"
    assert ksize[0] == 1 and ksize[3] == 1, "unsupported ksize"
    input_shape = get_shape(input)
    batch_size = input_shape[0]
    h = input_shape[1]
    w = input_shape[2]
    channel_num = input_shape[3]

    input = tf.transpose(
        input, [3, 0, 1, 2],
        name=name + "/input_perm" if name is not None else None
    )
    input = tf.reshape(
        input, [channel_num*batch_size, h, w, 1],
        name=name + "/input_reshape" if name is not None else None
    )

    ksize = [ksize[1], ksize[2], 1, 1]
    kernel_elt_count = np.prod(np.array(ksize))
    kfilter = tf.ones(ksize, dtype=input.dtype) / kernel_elt_count
    output = tf.nn.conv2d(
        input, kfilter, strides=strides, padding=padding, 
        name=name + "/pool" if name is not None else None
    )

    output_shape = get_shape(output)
    oh = output_shape[1]
    ow = output_shape[2]

    output = tf.reshape(
        output, [channel_num, batch_size, oh, ow],
        name=name + "/output_reshape" if name is not None else None
    )
    output = tf.transpose(
        output, [1, 2, 3, 0],
        name=name + "/output_reshape" if name is not None else None
    )

    return output


average_pool = average_pool_v1


def switch_pool(input, pooling_switches, name=None):
    pooled_shape = get_shape(pooling_switches)
    batch_size = pooled_shape[0]
    element_num = np.prod(pooled_shape[1:])
    assert get_shape(input)[0]==batch_size, \
        "mismatched batch_size"
    # globalize indexes over samples
    global_base = np.reshape(np.array(range(batch_size))*element_num, [batch_size]+[1]*(len(pooled_shape)-1))
    pooling_switches += tf.constant(
        global_base, dtype=pooling_switches.dtype,
        name=name + "/baseswitches" if name is not None else None
    )
    flatten_switches = tf.reshape(
        pooling_switches, [-1],
        name=name + "/flattenswitches" if name is not None else None
    )
    flatten_input = tf.reshape(
        input, [-1],
        name=name + "/flatteninput" if name is not None else None
    )
    flatten_output = tf.gather(
        flatten_input, flatten_switches,
        name=name + "/flattenoutput" if name is not None else None
    )
    pooled_output = tf.reshape(
        flatten_output, pooled_shape,
        name=name + "/pooledoutput" if name is not None else None
    )
    return pooled_output


def max_pool_v1(input, ksize, strides, padding, alpha=0.1, name=None):
    _, pooling_switches = tf.nn.max_pool_with_argmax(
        input, ksize, strides, padding, tf.int64,
        name=name+"/switches" if name is not None else None)
    pooling_switches = tf.stop_gradient(
        pooling_switches,
        name=name + "/stop_gradients" if name is not None else None
    )
    max_pool_output = switch_pool(input, pooling_switches, name=name)
    return max_pool_output


def max_pool_v2(input, ksize, strides, padding, alpha=0.1, name=None):
    _, pooling_switches = tf.nn.max_pool_with_argmax(
        input, ksize, strides, padding, tf.int64,
        name=name+"/switches" if name is not None else None)
    pooling_switches = tf.stop_gradient(
        pooling_switches,
        name=name + "/stop_gradients" if name is not None else None
    )
    max_pool_output = switch_pool(input, pooling_switches, name=name)
    avg_pool_output = average_pool(
        input, ksize, strides, padding, name=name + "/avg_pool" if name is not None else None
    )
    return alpha*avg_pool_output + (1-alpha)*max_pool_output


def softmax_pool(input, ksize, strides, padding, name=None):

    assert strides[0] == 1 and strides[3] == 1, "unsupported strides"
    assert ksize[0] == 1 and ksize[3] == 1, "unsupported ksize"

    # ksize = [ksize[1], ksize[2], 1, 1]

    input_shape = get_shape(input)
    batch_size = input_shape[0]
    channel_num = input_shape[3]

    imcols = tf.extract_image_patches(
        input, ksize, strides, rates=[1, 1, 1, 1], padding=padding,
        name=name + "/patches" if name is not None else None
    )

    imcols_shape = get_shape(imcols)
    oh = imcols_shape[1]
    ow = imcols_shape[2]
    patch_size = imcols_shape[3]//channel_num
    imcols = tf.reshape(
        imcols, [batch_size, oh, ow, patch_size, channel_num],
        name=name + "/reshape1" if name is not None else None
    )

    patch_elt_weights = tf.nn.softmax(
        imcols, 3,
        name=name + "/softmax" if name is not None else None
    )
    patch_elt_weights = tf.stop_gradient(
        patch_elt_weights, name=name + "/stop_gradients" if name is not None else None
    )
    pooled_output = tf.reduce_sum(
        patch_elt_weights * imcols, axis=-2,
        name=name + "/sum" if name is not None else None
    )

    return pooled_output


max_pool = softmax_pool


def dropout(x, keep_prob, alpha=0.1, noise_shape=None, seed=None, name=None):
    x_shape = get_shape(x)
    if noise_shape is None:
        noise_shape = x_shape
    tf.set_random_seed(seed)
    noise = tf.random_uniform(
        noise_shape, minval=0., maxval=1., dtype=tf.float16, seed=seed,
        name=name+"_rand" if name is not None else None
    )
    kept_idxb = noise < keep_prob
    if not np.all(np.array(x_shape) == np.array(noise_shape)):
        kept_idxb = tf.tile(
            kept_idxb, (np.array(x_shape) // np.array(noise_shape)).tolist(),
            name=name+"_rand_tile" if name is not None else None
        )

    kept_idxb = tf.stop_gradient(
        kept_idxb, name=name+"_stop_gradients" if name is not None else None
    )
    # y = tf.where(kept_idxb, x, tf.zeros_like(x), name=name) * (1./keep_prob)
    total_weight = keep_prob+alpha*(1.-keep_prob)
    y = tf.where(kept_idxb, x, alpha*x, name=name) * (1./total_weight)

    return y


def normalized_index_template(n, dtype=tf.float32, offset=0):
    index_tpl = tf.constant(
        np.array(
            (np.arange(n)+offset) / n,
            dtype=dtype.as_numpy_dtype),
        dtype=dtype
    )
    return index_tpl


def random_truncated_masking(x, rand_func=None):

    if rand_func is None:
        def tf_uniform_random(num):
            return tf.random_uniform([num], minval=0., maxval=1.)
        rand_func = tf_uniform_random

    input_shape = get_shape(x)
    batch_size = input_shape[0]
    input_channels = input_shape[-1]
    input_rank = len(input_shape)

    random_mask_trunc = rand_func(batch_size)
    random_mask_trunc = tf.reshape(random_mask_trunc, [batch_size] + [1]*(input_rank-1))
    index_tpl = normalized_index_template(input_channels, dtype=x.dtype)
    index_tpl = tf.reshape(index_tpl, [1]*(input_rank-1) + [input_channels])
    chosen_idxb = (index_tpl <= random_mask_trunc)

    masked_x = tf.where(chosen_idxb, x, tf.zeros_like(x))

    return masked_x


def gaussian_masking_1d(mu, sample_num, sigma=1.):
    x = normalized_index_template(sample_num, dtype=mu.dtype)
    x = tf.reshape(x, [1]*len(get_shape(mu)) + [sample_num])
    mu = tf.expand_dims(mu, axis=-1)
    y = tf.exp(-0.5*tf.square((x-mu)/sigma))/(math.sqrt(2*math.pi)*sigma)
    return y


def gaussian_masking_2d(mu, sample_num, sigma=1.):
    mu_shape = get_shape(mu)
    assert mu_shape[-1] == 2, "mu must have two channels"

    if isinstance(sample_num, (tuple, list)):
        assert len(sample_num)==2, "sample_num's len must be 2"
        if sample_num[0] == sample_num[1]:
            sample_num = sample_num[0]

    if isinstance(sigma, (tuple, list)):
        assert len(sigma) ==2 , "sigma's len must be 2"
        if sigma[0] == sigma[1]:
            sigma = sigma[0]

    if not isinstance(sample_num, (tuple, list)) and not isinstance(sigma, (tuple, list)):
        p_1d = gaussian_masking_1d(mu, sample_num, sigma=sigma)
        r, c = tf.split(p_1d, 2, axis=len(mu_shape)-1)
    else:
        if not isinstance(sample_num, (tuple, list)):
            sample_num = [sample_num, sample_num]
        if not isinstance(sigma, (tuple, list)):
            sigma = [sigma, sigma]
        mu_r, mu_c = tf.split(mu, 2, axis=len(mu_shape)-1)
        r = gaussian_masking_1d(mu_r, sample_num[0], sigma[0])
        c = gaussian_masking_1d(mu_c, sample_num[1], sigma[1])

    r = tf.squeeze(r, axis=len(mu_shape)-1)
    c = tf.squeeze(c, axis=len(mu_shape)-1)

    r = tf.expand_dims(r, axis=-1)
    c = tf.expand_dims(c, axis=-2)
    m = tf.matmul(r, c)

    return m


def binominal_kl_divergence(p, q, is_p_gt=None):
    if is_p_gt is None:
        is_p_gt = not is_tf_data(p)
    if is_p_gt:
        if p == 0:
            kl = -tf.log(1.-q+epsilon)
        elif p == 1:
            kl = -tf.log(q+epsilon)
        else:
            kl = p * (tf.log(p) - tf.log(q + epsilon)) + (1 - p) * (tf.log(1. - p) - tf.log(1. - q + epsilon))
    else:
        kl = p * (tf.log(p+epsilon) - tf.log(q+epsilon)) + (1-p) * (tf.log(1.-p+epsilon) - tf.log(1.-q+epsilon))
    return kl


def diag_to_matrix(d):
    # support batch mode
    s = get_shape(d)
    if s[-1] == 0:
        return tf.zeros(s+[0], dtype=d.dtype)
    if s[-1] == 1:
        return tf.expand_dims(d)
    r = len(s)
    q = tf.split(d, s[-1], axis=r-1)
    z = tf.zeros(s, dtype=d.dtype)
    l = [q[0]]
    for i in range(1, s[-1]):
        l.append(z)
        l.append(q[i])
    m = tf.concat(l, axis=r-1)
    m = tf.reshape(m, s[:-1] + [s[-1], s[-1]])
    return m


def pad(tensor, paddings, mode='CONSTANT', name=None, *args, **kwargs):
    if mode == "MEAN_EDGE":
        if args:
            geometric_axis = args[0]
            args = args[1:]
        elif "geometric_axis" in kwargs:
                geometric_axis = kwargs["geometric_axis"]
                del kwargs["geometric_axis"]
        else:
            geometric_axis = None
        assert not args and not kwargs, "Unrecongized arguments"
        return pad_mean_edge(tensor, paddings, geometric_axis)
    else:
        output = tf.pad(tensor, paddings, mode, name, *args, **kwargs)
    return output


def pad_mean_edge(tensor, paddings, geometric_axis, name=None):

    assert geometric_axis, "at one axis as geometric axis"

    tensor = tf.convert_to_tensor(tensor)

    s = tf.shape(tensor)
    d = get_shape(s)[0]

    if geometric_axis is None:
        geometric_axis = list(range(d))

    v = 0
    n = 0
    m = 1
    for i in geometric_axis:
        m *= s[i]

    if name is None:
        name = "pad_mean_edge"

    with tf.name_scope(name):
        paddings = tf.convert_to_tensor(paddings)

        mask = tf.pad(tf.ones_like(tensor, dtype=tf.bool), paddings, mode='CONSTANT', constant_values=False)
        val_tensor = tf.pad(tensor, paddings, mode='CONSTANT')

        geometric_axis_b = [False] * d
        for i in geometric_axis:
            ind = [slice(None, None)] * d
            ind[i] = slice(0, 1)
            e1 = tensor[ind]
            ind[i] = slice(-1, None)
            e2 = tensor[ind]
            n += m // s[i] * 2
            v += tf.reduce_sum(tf.concat([e1, e2], axis=i), axis=geometric_axis, keep_dims=True)
            geometric_axis_b[i] = True
        v /= tf.cast(n, dtype=v.dtype)

        r = tf.shape(mask)
        rep = tf.where(geometric_axis_b, r, tf.ones_like(r))

        pad_tensor = tf.tile(v, rep)
        output = tf.where(mask, val_tensor, pad_tensor)

    return output


def stop_gradient_if_not(cond, *args):
    outputs = []
    for v in args:
        outputs.append(tf.reshape(tf.where(cond, v, tf.stop_gradient(v)), get_shape(v)))
    if len(outputs) == 0:
        outputs = None
    elif len(outputs) == 1:
        outputs = outputs[0]
    else:
        outputs = tuple(outputs)
    return outputs

