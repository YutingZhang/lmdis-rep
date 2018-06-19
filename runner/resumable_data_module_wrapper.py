from zutils.py_utils import link_with_instance


class Net:

    def __init__(self, data_module, _num_sample_factors=None):

        if _num_sample_factors is None:
            self._num_sample_factors = 1
        else:
            self._num_sample_factors = _num_sample_factors
        self._num_samples = data_module.num_samples()
        self.data_module = data_module
        self._total_pos = \
            data_module.num_samples_finished() + \
            self._num_samples * data_module.epoch()
        self._total_iter = data_module.iter()

        link_with_instance(self, self.data_module)

    def __call__(self, *args, **kwargs):
        return self.next_batch(*args, **kwargs)

    def next_batch(self, batch_size):
        self._total_iter += 1
        self._total_pos += batch_size
        data = self.data_module(batch_size)
        return data

    def num_samples_finished(self):
        return (self._total_pos % self._num_samples) * self._num_sample_factors

    def epoch(self):
        return (self._total_pos // self._num_samples) * self._num_sample_factors

    def iter(self):
        return self._total_iter * self._num_sample_factors

    def pos(self):
        return self._total_pos * self._num_sample_factors

    def set_iter(self, new_iter):
        self._total_iter = new_iter * self._num_sample_factors

    def set_pos(self, new_pos):
        self._total_pos = new_pos * self._num_sample_factors

    def reset(self):
        self._total_pos = 0
        self._total_iter = 0

    def num_samples(self):
        return self._num_samples * self._num_sample_factors

