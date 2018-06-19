from abc import ABCMeta, abstractmethod
import tensorflow as tf
from collections import OrderedDict
import math
import zutils.tf_math_funcs as tmf

epsilon = tmf.epsilon


class Factory:

    def sampling(self, dist_param, batch_size, latent_dim):
        """ Create network for VAE latent variables (sampling only)

        :param dist_param: input for the posterior
        :param batch_size: batch size
        :param latent_dim: dimension of the latent_variables
        :return: samples - random samples from either posterior or prior distribution
        """

        # generate random samples
        rho = tf.random_uniform([batch_size, latent_dim])
        return self.inv_cdf(dist_param, rho)

    @abstractmethod
    def nll(self, dist_param, samples):
        return None, None

    def nll_formatted(self, dist_param, samples):
        x, a = self.nll(dist_param, samples)
        if isinstance(a, bool):
            s = x.get_shape()
            a = tf.tile(tf.reshape(tf.constant(a), [1]*len(s)), s)
        return x, a
