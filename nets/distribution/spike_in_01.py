import tensorflow as tf
import zutils.tf_math_funcs as tmf
import nets.distribution.spike

BaseFactory = nets.distribution.spike.Factory

epsilon = tmf.epsilon


class Factory(BaseFactory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def transform2param(cls, input_tensor, latent_dim):
        """ Create network for converting input_tensor to distribution parameters

        :param input_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: param_tensor - distribution parameters
        """
        param_tensor = tf.sigmoid(input_tensor)
        return param_tensor
