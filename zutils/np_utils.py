import numpy as np


def onthot_to_int(onehot_array, axis=1, dtype=np.int64, keepdims=False):

    s = onehot_array.shape
    num = s[axis]
    nonzero_indexes = np.nonzero(onehot_array)
    index_arr = np.array(np.arange(num))
    all_indexes = index_arr[nonzero_indexes[axis]]
    if keepdims:
        s[axis] = 1
    else:
        s = s[0:axis] + s[axis+1:]
    all_indexes = np.reshape(all_indexes, s)

    return all_indexes
