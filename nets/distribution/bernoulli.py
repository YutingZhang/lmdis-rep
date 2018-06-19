import tensorflow as tf
import numpy as np
from collections import OrderedDict
import math
import zutils.tf_math_funcs as tmf
import nets.distribution.generic
import nets.distribution.category

CategoryFactory = nets.distribution.category.Factory

GenericFactory = nets.distribution.generic.Factory

epsilon = tmf.epsilon


class Factory(GenericFactory):

    def __init__(self, tau=0., **kwargs):
        self.categ_dist = CategoryFactory(pi=[0.5, 0.5], tau=tau)

    @staticmethod
    def param_num():
        return 1

    @staticmethod
    def param_dict(p=0.5):
        return OrderedDict(p=p)

    @classmethod
    def transform2param(cls, input_tensor, latent_dim):
        """ Create network for converting input_tensor to distribution parameters

        :param input_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: param_tensor - distribution parameters
        """
        param_tensor = tf.sigmoid(input_tensor)
        return param_tensor

    @classmethod
    def parametrize(cls, param_tensor, latent_dim):
        """ Create network for converting parameter_tensor to parameter dictionary

        :param param_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: dist_param - distribution parameters
        """
        dist_param = cls.param_dict(
            p=param_tensor,
        )
        return dist_param

    @classmethod
    def deparametrize(cls, dist_param):
        param_tensor = tmf.expand_minimum_ndim(dist_param["p"], axis=0)
        return param_tensor

    def nll(self, dist_param, samples):
        """ Compute negative log likelihood on given sample and distribution parameter

        :param samples: samples for evaluating PDF
        :param dist_param: input for the posterior
        :return: likelihood - likelihood to draw such samples from given distribution
        :return: is_atomic - is atomic, scalar or the same size as likelihood
        """
        likelihood = tf.where(samples > 0.5, dist_param["p"], 1.0-dist_param["p"])
        bernoulli_nll = -tf.log(likelihood+epsilon)
        return bernoulli_nll, True

    def sampling(self, dist_param, batch_size, latent_dim):
        """ Create network for VAE latent variables (sampling only)

        :param dist_param: input for the posterior
        :param batch_size: batch size
        :param latent_dim: dimension of the latent_variables
        :return: samples - random samples from either posterior or prior distribution
        """

        # generate random samples
        if self.categ_dist.tau>0.0:
            # soft sampling
            p = self.deparametrize(dist_param)
            categ_dist_param = OrderedDict()
            categ_dist_param["K"] = 2
            t_p = tf.reshape(p, [batch_size, 1, latent_dim])
            categ_dist_param["pi"] = tf.concat([1.0-t_p, t_p], axis=1)
            categ_samples = self.categ_dist.sampling(dist_param, batch_size, latent_dim)
            return categ_samples[:,1]
        else:
            # hard sampling
            rho = tf.random_uniform([batch_size, latent_dim])
            return self.inv_cdf(dist_param, rho)

    def inv_cdf(self, dist_param, rho):
        p = self.deparametrize(dist_param)
        return tf.to_float(rho > 1.0-p)

    @staticmethod
    def self_entropy(dist_param):
        p = dist_param["p"]
        se = -p*tf.log(p+epsilon) - (1.0-p)*tf.log(1.0-p+epsilon)
        return se

    @classmethod
    def kl_divergence(cls, dist_param, ref_dist_param, ref_dist_type=None):
        if not isinstance(ref_dist_type, cls) and ref_dist_type is not None:   # handle hybrid distribution
            return None

        p = dist_param["p"]
        p0 = ref_dist_param["p"]
        homo_kl = p*tf.log(p/(p0+epsilon)+epsilon) + (1.0-p)*tf.log((1.0-p)/(1.0-p0+epsilon)+epsilon)
        return homo_kl

    @classmethod
    def cross_entropy(cls, dist_param, ref_dist_param, ref_dist_type=None):
        if not isinstance(ref_dist_type, cls) and ref_dist_type is not None:   # handle hybrid distribution
            return None

        p = dist_param["p"]
        p0 = ref_dist_param["p"]
        homo_ce = -p*tf.log(tf.clip_by_value(p0+epsilon, clip_value_min=epsilon, clip_value_max=1.)) - \
                  (1.0-p)*tf.log(tf.clip_by_value(1.0-p0+epsilon, clip_value_min=epsilon, clip_value_max=1.))
        return homo_ce

    @staticmethod
    def mean(dist_param):
        p = dist_param["p"]
        return p
