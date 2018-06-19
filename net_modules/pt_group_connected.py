import collections

import tensorflow as tf

from prettytensor import layers
from prettytensor import parameters
from prettytensor import pretty_tensor_class as prettytensor
from prettytensor.pretty_tensor_class import PROVIDED

# pylint: disable=invalid-name
@prettytensor.Register(assign_defaults=('activation_fn', 'l2loss',
                                        'parameter_modifier', 'phase'))
class group_connected(prettytensor.VarStoreMethod):

  def __call__(
        self,
        input_layer,
        size,
        activation_fn=None,
        l2loss=None,
        weights=None,
        bias=tf.zeros_initializer(),
        transpose_weights=False,
        phase=prettytensor.Phase.train,
        parameter_modifier=parameters.identity,
        tie_groups=False,
        name=PROVIDED
  ):
    """Adds the parameters for a fully connected layer and returns a tensor.
    The current PrettyTensor must have rank 2.
    Args:
      input_layer: The Pretty Tensor object, supplied.
      size: The number of neurons
      activation_fn: A tuple of (activation_function, extra_parameters). Any
        function that takes a tensor as its first argument can be used. More
        common functions will have summaries added (e.g. relu).
      l2loss: Set to a value greater than 0 to use L2 regularization to decay
        the weights.
      weights:  An initializer for weights or a Tensor. If not specified,
        uses He's initialization.
      bias: An initializer for the bias or a Tensor. No bias if set to None.
      transpose_weights: Flag indicating if weights should be transposed;
        this is useful for loading models with a different shape.
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
    if input_layer.get_shape().ndims != 3:
      raise ValueError(
          'group_connected requires a rank 3 Tensor with known 2nd and 3rd '
          'dimension: %s' % input_layer.get_shape())
    group_num = input_layer.shape[1]
    in_size = input_layer.shape[2]
    if group_num is None:
      raise ValueError('Number of groups must be known.')
    if in_size is None:
      raise ValueError('Number of input nodes must be known.')
    books = input_layer.bookkeeper
    if weights is None:
      weights = layers.he_init(in_size, size, activation_fn)

    dtype = input_layer.tensor.dtype
    weight_shape = [group_num, size, in_size] if transpose_weights else [group_num, in_size, size]

    params_var = parameter_modifier(
        'weights',
        self.variable('weights', weight_shape,
                      weights, dt=dtype),
        phase)

    if tie_groups and phase == prettytensor.Phase.train:
        with tf.variable_scope("weight_tying"):
            params = tf.tile(tf.reduce_mean(params_var, axis=0, keep_dims=True), [group_num, 1, 1])
            with tf.control_dependencies([tf.assign(params_var, params)]):
                params = tf.identity(params)
    else:
        params = params_var

    input_tensor = tf.expand_dims(input_layer, axis=-2)
    params_tensor = tf.tile(tf.expand_dims(params, axis=0), [tf.shape(input_tensor)[0], 1, 1, 1])
    y = tf.matmul(input_tensor, params_tensor, transpose_b=transpose_weights, name=name)
    y = tf.squeeze(y, axis=2)
    layers.add_l2loss(books, params, l2loss)
    if bias is not None:
      y += parameter_modifier(
          'bias',
          self.variable('bias', [size], bias, dt=dtype),
          phase)

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


def main():
    import prettytensor as pt
    input_tensor = tf.zeros([64, 5, 7])
    output_tensor = pt.wrap(input_tensor).group_connected(15).tensor
    print(output_tensor)


if __name__ == "main":
    main()
