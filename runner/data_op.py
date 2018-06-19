import tensorflow as tf
from operator import itemgetter


class RecordedDataFetcher:
    def __init__(self, data_mod, batch_size, debug_text=None):
        self.iter = 0
        self.data_mod = data_mod
        self.batch_size = batch_size
        self.debug_text = debug_text

    def next_batch(self):
        self.iter += 1
        if self.debug_text is not None:
            print("%s [iter %d]" % (str(self.debug_text), self.iter))
        return self.data_mod(self.batch_size)


def tf_variable_from_data_module(*args, **kwargs):
    with tf.device("/cpu:0"):
        return _tf_variable_from_data_module(*args, **kwargs)


def _tf_variable_from_data_module(data_mod, batch_size, output_index=0, debug_text=None):
    out_type = data_mod.output_types()
    out_shape = data_mod.output_shapes()
    for i in range(len(out_type)):
        if isinstance(out_type[i], str):
            out_type[i] = getattr(tf, out_type[i])

    rdf = RecordedDataFetcher(data_mod, batch_size, debug_text)

    data = tf.py_func(rdf.next_batch, [], out_type)
    if not (isinstance(data, list) or isinstance(data, tuple)):
        out_shape = list(out_shape)
        out_shape[0] = batch_size
        data = tf.reshape(data, out_shape)
    else:
        data = list(data)
        out_shape = list(out_shape)
        for i in range(len(out_shape)):
            out_shape[i] = list(out_shape[i])
            out_shape[i][0] = batch_size
            data[i] = tf.reshape(data[i], out_shape[i])

    assert output_index is not None, "output_index should be a scalar or a list"
    if not (isinstance(data, list) or isinstance(data, tuple)):
        data = [data]

    for i in range(len(data)):
        data[i] = tf.stop_gradient(data[i])

    out_keys = data_mod.output_keys()
    data_dict = dict()
    for i in range(len(data)):
        data_dict[i] = data[i]
        data_dict[out_keys[i]] = data[i]

    if isinstance(output_index, (list, tuple)):
        return itemgetter(*output_index)(data_dict)
    else:
        return itemgetter(output_index)(data_dict)
