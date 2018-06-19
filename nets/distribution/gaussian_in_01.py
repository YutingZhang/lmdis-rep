import tensorflow as tf
import zutils.tf_math_funcs as tmf
import nets.distribution.gaussian

BaseFactory = nets.distribution.gaussian.Factory

epsilon = tmf.epsilon


class Factory(BaseFactory):

    default_stddev_lower_bound = 0.05

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def transform2param(cls, input_tensor, latent_dim):
        """ Create network for converting input_tensor to distribution parameters

        :param input_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: param_tensor - distribution parameters
        """
        param_tensor = tf.concat(
            [tf.sigmoid(input_tensor[:, :latent_dim]),
             tmf.atanh_sigmoid(input_tensor[:, latent_dim:])+epsilon],
            axis=1
        )
        return param_tensor

    @staticmethod
    def sample_to_real(samples):
        return tmf.inv_sigmoid(samples)

    @staticmethod
    def real_to_samples(samples_in_real):
        return tf.sigmoid(samples_in_real)
