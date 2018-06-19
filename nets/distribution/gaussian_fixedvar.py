import tensorflow as tf
from collections import OrderedDict
import math
import zutils.tf_math_funcs as tmf
import nets.distribution.generic

GenericFactory = nets.distribution.generic.Factory

epsilon = tmf.epsilon


class Factory(GenericFactory):

    def __init__(self, **kwargs):
        if "options" in kwargs and "stddev" in kwargs["options"]:
            self.stddev = kwargs["options"]["stddev"]
        else:
            self.stddev = 1.

    @staticmethod
    def param_num():
        return 1

    @staticmethod
    def param_dict(mean=0.0):
        return OrderedDict(mean=mean)

    @classmethod
    def transform2param(cls, input_tensor, latent_dim):
        """ Create network for converting input_tensor to distribution parameters

        :param input_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: param_tensor - distribution parameters
        """
        assert tmf.get_shape(input_tensor)[1] == latent_dim, "wrong dim"
        param_tensor = input_tensor
        return param_tensor

    @classmethod
    def parametrize(cls, param_tensor, latent_dim):
        """ Create network for converting parameter_tensor to parameter dictionary

        :param param_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: dist_param - distribution parameters
        """
        assert tmf.get_shape(param_tensor)[1] == latent_dim, "wrong dim"
        dist_param = cls.param_dict(
            mean=param_tensor,
        )
        return dist_param

    @classmethod
    def deparametrize(cls, dist_param):
        param_tensor = tmf.expand_minimum_ndim(dist_param["mean"], 2)
        return param_tensor

    def nll(self, dist_param, samples):
        """ Compute negative log likelihood on given sample and distribution parameter

        :param samples: samples for evaluating PDF
        :param dist_param: input for the posterior
        :return: likelihood - likelihood to draw such samples from given distribution
        :return: is_atomic - is atomic, scalar or the same size as likelihood
        """
        u = dist_param["mean"]
        s = self.stddev
        x = samples

        gaussian_nll = 0.5*(tf.square((x-u)/s) +
                            2.0*tf.log(s) + math.log(2.0*math.pi))

        x = tf.check_numerics(x, "gaussian nll inf or nan", "gaussian_nll_check")

        return gaussian_nll, False

    def sampling(self, dist_param, batch_size, latent_dim):
        """ Create network for VAE latent variables (sampling only)

        :param dist_param: input for the posterior
        :param batch_size: batch size
        :param latent_dim: dimension of the latent_variables
        :return: samples - random samples from either posterior or prior distribution
        """

        # generate random samples
        varepsilon = tf.random_normal([batch_size, latent_dim])
        samples = dist_param["mean"] + varepsilon * self.stddev
        return samples

    """
    @staticmethod
    def inv_cdf(dist_param, rho):
        pass
    """

    def self_entropy(self, dist_param):
        s = self.stddev
        gaussian_entropy = 0.5*(math.log(2*math.pi)+1.0) + tf.log(s + epsilon)

        gaussian_entropy = tf.ones_like(dist_param["mean"]) * gaussian_entropy

        return gaussian_entropy

    def kl_divergence(self, dist_param, ref_dist_param, ref_dist_type=None):
        if not isinstance(ref_dist_type, type(self)) and ref_dist_type is not None:   # handle hybrid distribution
            return None

        u0 = ref_dist_param["mean"]
        s0 = self.stddev
        u = dist_param["mean"]
        s = self.stddev

        homo_kl = (tf.square(s) + tf.square(u-u0) - 1.0) / (2*s0)

        return homo_kl

    def cross_entropy(self, dist_param, ref_dist_param, ref_dist_type=None):
        if not isinstance(ref_dist_type, type(self)) and ref_dist_type is not None:   # handle hybrid distribution
            return None

        u0 = ref_dist_param["mean"]
        s0 = self.stddev
        u = dist_param["mean"]
        s = self.stddev

        homo_ce = 0.5*(math.log(2*math.pi)+1.0) + tf.log(s0 + epsilon) + (tf.square(s) + tf.square(u-u0) - 1.0) / (2*s0)

        return homo_ce

    @staticmethod
    def mean(dist_param):
        mean = dist_param["mean"]
        return mean

    @staticmethod
    def sample_to_real(samples):
        return samples

    @staticmethod
    def real_to_samples(samples_in_real):
        return samples_in_real
