import numpy as np
import tensorflow as tf

import zutils.tf_math_funcs as tmf
from net_modules.gen import get_net_factory

epsilon = tmf.epsilon


class Factory:

    def __init__(self, latent_dist_name, *args, **kwargs):
        self.output_dist = get_net_factory("distribution", latent_dist_name, *args, **kwargs)
        assert self.output_dist is not None, "Cannot get the distribution"

    def __call__(self, input_tensor, gt_tensor):

        input_tensor, input_shape = self.flatten_dist_tensor(input_tensor)

        gt_s = tmf.get_shape(gt_tensor)
        gt_tensor = tf.reshape(gt_tensor, [gt_s[0], np.prod(gt_s[1:])])

        dist_param, _ = self.visible_dist(input_tensor)
        nll, _ = self.output_dist.nll(dist_param, gt_tensor)

        nll = tf.reshape(nll, input_shape)
        return nll

    def flatten_dist_tensor(self, dist_tensor):
        s = tmf.get_shape(dist_tensor)
        total_hidden = np.prod(s[1:])
        input_tensor = tf.reshape(dist_tensor, [s[0], total_hidden//self.param_num(), self.param_num()])
        input_tensor = tf.transpose(input_tensor, [0, 2, 1])    # move the channel in front of geometric axes
        input_tensor = tf.reshape(input_tensor, [s[0], total_hidden])
        s[-1] //= self.param_num()
        return input_tensor, s

    def self_entropy(self, input_tensor):

        input_tensor, input_shape = self.flatten_dist_tensor(input_tensor)

        dist_param, _ = self.visible_dist(input_tensor)
        se = self.output_dist.self_entropy(dist_param)

        se = tf.reshape(se, input_shape)
        return se

    def mean(self, input_tensor):

        input_tensor, input_shape = self.flatten_dist_tensor(input_tensor)

        dist_param, param_tensor = self.visible_dist(input_tensor)
        vis = self.output_dist.mean(dist_param)

        param_tensor = tf.reshape(
            param_tensor, [input_shape[0], self.param_num()]+input_shape[1:])
        vis = tf.reshape(vis, input_shape)
        return vis, param_tensor

    visualize = mean

    def visible_dist(self, input_tensor):
        s = tmf.get_shape(input_tensor)
        latent_dim = np.prod(s[1:])//self.param_num()
        param_tensor = self.output_dist.transform2param(input_tensor, latent_dim)
        dist_param = self.output_dist.parametrize(param_tensor, latent_dim)
        return dist_param, param_tensor

    def param_num(self):
        return self.output_dist.param_num()
