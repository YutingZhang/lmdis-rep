import tensorflow as tf
from collections import OrderedDict
import math
import zutils.tf_math_funcs as tmf
import nets.distribution.generic

GenericFactory = nets.distribution.generic.Factory

epsilon = tmf.epsilon


class Factory(GenericFactory):

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def is_atomic():    # True for discrete distribution
        return True

    @staticmethod
    def param_num():
        return 1

    @staticmethod
    def param_dict(a=0.0):
        return OrderedDict(a=a)

    def transform2param(self, input_tensor, latent_dim):
        """ Create network for converting input_tensor to distribution parameters

        :param input_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: param_tensor - distribution parameters
        """
        param_tensor = input_tensor
        return param_tensor

    @classmethod
    def parametrize(cls, param_tensor, latent_dim):
        """ Create network for converting parameter_tensor to parameter dictionary

        :param param_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: dist_param - distribution parameters
        """
        dist_param = cls.param_dict(
            a=param_tensor
        )
        return dist_param

    @classmethod
    def deparametrize(cls, dist_param):
        param_tensor = dist_param["a"]
        return param_tensor

    @staticmethod
    def nll(dist_param, samples):
        """ Compute negative log likelihood on given sample and distribution parameter

        :param samples: samples for evaluating PDF
        :param dist_param: input for the posterior
        :return: likelihood - likelihood to draw such samples from given distribution
        :return: is_atomic - is atomic, scalar or the same size as likelihood
        """
        spike_nll = tf.where(
            samples == dist_param["a"],
            tf.ones_like(samples), -tf.ones_like(samples)*math.inf)
        return spike_nll, True

    @staticmethod
    def sampling(dist_param, batch_size, latent_dim):
        """ Create network for VAE latent variables (sampling only)

        :param dist_param: input for the posterior
        :param batch_size: batch size
        :param latent_dim: dimension of the latent_variables
        :return: samples - random samples from either posterior or prior distribution
        """

        # generate random samples
        return tf.ones([batch_size, latent_dim]) * dist_param["a"]

    @staticmethod
    def self_entropy(dist_param):
        return tf.zeros_like(dist_param["a"])

    @classmethod
    def kl_divergence(cls, dist_param, ref_dist_param, ref_dist_type=None):
        if not isinstance(ref_dist_type, cls) and ref_dist_type is not None:   # handle hybrid distribution
            return None
        return tf.zeros_like(dist_param["a"])
        # return tf.where(
        #     dist_param["a"] == ref_dist_param["a"],
        #     tf.zeros_like(dist_param["a"]), tf.ones_like(dist_param["a"])*math.inf)

    @classmethod
    def cross_entropy(cls, dist_param, ref_dist_param, ref_dist_type=None):
        if not isinstance(ref_dist_type, cls) and ref_dist_type is not None:   # handle hybrid distribution
            return None
        return tf.zeros_like(dist_param["a"])
        # return tf.where(
        #     dist_param["a"] == ref_dist_param["a"],
        #     tf.zeros_like(dist_param["a"]), tf.ones_like(dist_param["a"])*math.inf)

    @staticmethod
    def mean(dist_param):
        return dist_param["a"]

    @staticmethod
    def sample_to_real(samples):
        return samples

    @staticmethod
    def real_to_samples(samples_in_real):
        return samples_in_real
