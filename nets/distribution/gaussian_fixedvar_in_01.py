import tensorflow as tf
import zutils.tf_math_funcs as tmf
import nets.distribution.gaussian_fixedvar

BaseFactory = nets.distribution.gaussian_fixedvar.Factory

epsilon = tmf.epsilon


class Factory(BaseFactory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "options" in kwargs and "pre_sigmoid" in kwargs["options"]:
            self.pre_sigmoid = kwargs["options"]["pre_sigmoid"]
        else:
            self.pre_sigmoid = False

    def transform2param(self, input_tensor, latent_dim):
        """ Create network for converting input_tensor to distribution parameters

        :param input_tensor: (posterior phase) input tensor for the posterior
        :param latent_dim: dimension of the latent_variables
        :return: param_tensor - distribution parameters
        """
        param_tensor = tf.concat(
            [input_tensor[:, :latent_dim] if self.pre_sigmoid else tf.sigmoid(input_tensor[:, :latent_dim]),
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
