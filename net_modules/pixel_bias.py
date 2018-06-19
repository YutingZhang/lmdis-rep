import tensorflow as tf

import collections

from prettytensor import layers
from prettytensor import parameters
from prettytensor import pretty_tensor_class as prettytensor
from prettytensor.pretty_tensor_class import PROVIDED


@prettytensor.Register(assign_defaults=('activation_fn', 'parameter_modifier', 'phase'))
class pixel_bias(prettytensor.VarStoreMethod):

    def __call__(
            self, input_layer, activation_fn=None, bias=tf.zeros_initializer(), phase=prettytensor.Phase.train,
            parameter_modifier=parameters.identity, name=PROVIDED
    ):
        """
        Adds the parameters for a fully connected layer and returns a tensor.
        The current PrettyTensor must have rank 2.
        Args:
          input_layer: The Pretty Tensor object, supplied.
          size: The number of neurons
          bias: An initializer for the bias or a Tensor. No bias if set to None.
          phase: The phase of graph construction.  See `pt.Phase`.
          parameter_modifier: A function to modify parameters that is applied after
            creation and before use.
          name: The name for this operation is also used to create/find the
            parameter variables.
        Returns:
          A Pretty Tensor handle to the layer.
        Raises:
          ValueError: if the Pretty Tensor is not rank 2  or the number of input
            nodes (second dim) is not known.
        """

        if input_layer.get_shape().ndims != 4:
            raise ValueError(
                'pixel_bias requires a rank 4 Tensor with known second '
                'dimension: %s' % input_layer.get_shape())
        if input_layer.shape[1] is None or input_layer.shape[2] is None or input_layer.shape[3] is None:
            raise ValueError('input size must be known.')

        x = input_layer.tensor
        dtype = input_layer.dtype
        books = input_layer.bookkeeper
        b = parameter_modifier(
            'bias',
            self.variable('bias', input_layer.shape[2:], bias, dt=dtype),
            phase)
        y = x + tf.expand_dims(b, axis=0)

        if activation_fn is not None:
            if not isinstance(activation_fn, collections.Sequence):
                activation_fn = (activation_fn,)
            y = layers.apply_activation(books,
                                        y,
                                        activation_fn[0],
                                        activation_args=activation_fn[1:])
        books.add_histogram_summary(y, '%s/activations' % y.op.name)

        return input_layer.with_tensor(y, parameters=self.vars)

# pylint: enable=invalid-name
