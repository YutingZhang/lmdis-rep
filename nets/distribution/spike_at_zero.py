import tensorflow as tf
from collections import OrderedDict
import math
import zutils.tf_math_funcs as tmf
import nets.distribution.generic

GenericFactory = nets.distribution.generic.Factory

epsilon = tmf.epsilon


class Factory(GenericFactory):

    def __init__(self):
        pass

    @staticmethod
    def is_atomic():    # True for discrete distribution
        return True

    @staticmethod
    def param_num():
        return 0

    @staticmethod
    def param_dict():
        return OrderedDict()

    @staticmethod
    def transform2param(input_tensor, latent_dim):
        """ Create network for converting input_tensor to distribution parameters

        :param input_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: param_tensor - distribution parameters
        """
        return input_tensor

    @classmethod
    def parametrize(cls, param_tensor, latent_dim):
        """ Create network for converting parameter_tensor to parameter dictionary

        :param param_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: dist_param - distribution parameters
        """
        dist_param = cls.param_dict()
        return dist_param

    @classmethod
    def deparametrize(cls, dist_param):
        param_tensor = None
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
            samples == tf.zeros_like(samples),
            tf.ones_like(samples), -tf.ones_like(samples) * math.inf)
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
        return tf.zeros([batch_size, latent_dim])

    @staticmethod
    def self_entropy(dist_param):
        return 0.0

    @classmethod
    def kl_divergence(cls, dist_param, ref_dist_param, ref_dist_type=None):
        if not isinstance(ref_dist_type, cls) and ref_dist_type is not None:   # handle hybrid distribution
            return None
        return 0.0

    @classmethod
    def cross_entropy(cls, dist_param, ref_dist_param, ref_dist_type=None):
        if not isinstance(ref_dist_type, cls) and ref_dist_type is not None:   # handle hybrid distribution
            return None
        return 0.0

    @staticmethod
    def mean(dist_param):
        return 0.0
