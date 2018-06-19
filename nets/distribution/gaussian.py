import tensorflow as tf
from collections import OrderedDict
import math
import zutils.tf_math_funcs as tmf
import nets.distribution.generic

GenericFactory = nets.distribution.generic.Factory

epsilon = tmf.epsilon


class Factory(GenericFactory):

    default_stddev_lower_bound = epsilon

    def __init__(self, **kwargs):
        if "options" in kwargs:
            self.options = kwargs["options"]
        else:
            self.options = dict()
        if "stddev_lower_bound" not in self.options:
            self.options["stddev_lower_bound"] = epsilon

    @staticmethod
    def param_num():
        return 2

    @staticmethod
    def param_dict(mean=0.0, stddev=1.0):
        return OrderedDict(mean=mean, stddev=stddev)

    def transform2param(self, input_tensor, latent_dim):
        """ Create network for converting input_tensor to distribution parameters

        :param input_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: param_tensor - distribution parameters
        """
        param_tensor = tf.concat(
            [input_tensor[:, :latent_dim],
             tf.maximum(tmf.atanh_sigmoid(input_tensor[:, latent_dim:]), self.options["stddev_lower_bound"])],
            axis=1
        )
        return param_tensor

    @classmethod
    def parametrize(cls, param_tensor, latent_dim):
        """ Create network for converting parameter_tensor to parameter dictionary

        :param param_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: dist_param - distribution parameters
        """
        dist_param = cls.param_dict(
            mean=param_tensor[:, :latent_dim],
            stddev=param_tensor[:, latent_dim:]
        )
        return dist_param

    @classmethod
    def deparametrize(cls, dist_param):
        param_tensor = tf.concat(
            [tmf.expand_minimum_ndim(dist_param["mean"], 2),
             tmf.expand_minimum_ndim(dist_param["stddev"], 2)], axis=1)
        return param_tensor

    @staticmethod
    def nll(dist_param, samples):
        """ Compute negative log likelihood on given sample and distribution parameter

        :param samples: samples for evaluating PDF
        :param dist_param: input for the posterior
        :return: likelihood - likelihood to draw such samples from given distribution
        :return: is_atomic - is atomic, scalar or the same size as likelihood
        """
        u = dist_param["mean"]
        s = dist_param["stddev"]
        x = samples

        gaussian_nll = 0.5*(tf.square((x-u)/s) +
                            2.0*tf.log(s) + math.log(2.0*math.pi))

        return gaussian_nll, False

    @staticmethod
    def sampling(dist_param, batch_size, latent_dim):
        """ Create network for VAE latent variables (sampling only)

        :param dist_param: input for the posterior
        :param batch_size: batch size
        :param latent_dim: dimension of the latent_variables
        :return: samples - random samples from either posterior or prior distribution
        """

        # generate random samples
        varepsilon = tf.random_normal([batch_size, latent_dim])
        samples = dist_param["mean"] + varepsilon * dist_param["stddev"]
        return samples

    """
    @staticmethod
    def inv_cdf(dist_param, rho):
        pass
    """

    @staticmethod
    def self_entropy(dist_param):
        s = dist_param["stddev"]
        gaussian_entropy = 0.5*(math.log(2*math.pi)+1.0) + tf.log(s + epsilon)
        return gaussian_entropy

    @classmethod
    def kl_divergence(cls, dist_param, ref_dist_param, ref_dist_type=None):
        if not isinstance(ref_dist_type, cls) and ref_dist_type is not None:   # handle hybrid distribution
            return None

        u0 = ref_dist_param["mean"]
        s0 = ref_dist_param["stddev"]
        u = dist_param["mean"]
        s = dist_param["stddev"]

        homo_kl = tf.log(s0 + epsilon) - tf.log(s + epsilon) + \
            0.5*(tf.square(s) + tf.square(u-u0)) / tf.square(s0) - 0.5

        return homo_kl

    @classmethod
    def cross_entropy(cls, dist_param, ref_dist_param, ref_dist_type=None):
        if not isinstance(ref_dist_type, cls) and ref_dist_type is not None:   # handle hybrid distribution
            return None

        u0 = ref_dist_param["mean"]
        s0 = ref_dist_param["stddev"]
        u = dist_param["mean"]
        s = dist_param["stddev"]

        homo_ce = 0.5*(math.log(2*math.pi)+1.0) + tf.log(s0 + epsilon) + \
            0.5 * (tf.square(s) + tf.square(u - u0)) / tf.square(s0) - 0.5

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
