import numpy as np
import scipy.ndimage
from zutils.py_utils import link_with_instance
from copy import copy


class Net:

    def __init__(self, data_module, options, mode=None):
        assert isinstance(options, dict), "wrong options"
        self.data_module = data_module
        self.opt_dict = options

        if mode is None:
            mode = "deterministic"
        assert mode in ("deterministic", "random")
        if mode == "deterministic":
            self.use_random = False
        else:
            self.use_random = True

        self.shorter_edge_length = self._shorter_edge_length()
        self.center_patch_size = self._center_patch_size()
        self.crop_size = self._crop_size()
        self._data_field_index = self.data_module.output_keys().index('data')

        # output shape
        original_output_shape = self.data_module.output_shapes()
        self._output_shape = copy(original_output_shape)
        if self.crop_size is not None:
            target_shape = self.crop_size
        elif self.center_patch_size is not None:
            target_shape = self.center_patch_size
        else:
            target_shape = None
        if target_shape is not None:
            _data_shape = self._output_shape[self._data_field_index]
            self._output_shape[self._data_field_index] = (_data_shape[0],) + target_shape + (_data_shape[3],)
        self.image_color_scaling = self._image_color_scaling()

        link_with_instance(self, self.data_module)

    def __call__(self, *args, **kwargs):
        return self.next_batch(*args, **kwargs)

    def _shorter_edge_length(self):
        if "image_shorter_edge_length" not in self.opt_dict:
            return None
        a = self.opt_dict["image_shorter_edge_length"]
        if a is None:
            return None
        if self.use_random:
            if "image_shorter_edge_length_list" not in self.opt_dict:
                return [a]
            b = self.opt_dict["image_shorter_edge_length_list"]
            if b is None:
                return [a]
            if isinstance(b, range):
                b = list(b)
            return b
        else:
            return a

    def _center_patch_size(self):
        if "image_center_patch_size" not in self.opt_dict:
            return None
        s = self.opt_dict["image_center_patch_size"]
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return s, s
        assert len(s) == 2, "wrong center patch specification"
        w = s[0]
        h = s[1]
        return w, h

    def _crop_size(self):
        if "image_crop_size" not in self.opt_dict:
            return None
        s = self.opt_dict["image_crop_size"]
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return s, s
        assert len(s) == 2, "wrong crop size specification"
        w = s[0]
        h = s[1]
        return w, h

    def _image_color_scaling(self):
        if "image_color_scaling" not in self.opt_dict:
            return None
        return self.opt_dict["image_color_scaling"]

    #def _image_background_colors(self):
    #    if "image_background_color" not in self.opt_dict or self.opt_dict["image_background_color"] is None:
    #        return None, None
    #    assert "image_background_replace_color" in self.opt_dict and self.opt_dict["image_background_color"], \
    #        "must specify image_background_replace_color"
    #    return self.opt_dict["image_background_color"], self.opt_dict["image_background_color"]

    def output_shapes(self):
        return self._output_shape

    @staticmethod
    def robust_center_crop(im, crop_size):
        im_rank = len(im.shape)
        ch = crop_size[0]
        cw = crop_size[1]
        if ch > im.shape[0] or cw < im.shape[1]:
            h_beg = (ch - im.shape[0]) // 2
            h_end = ch - im.shape[0] - h_beg
            w_beg = (cw - im.shape[1]) // 2
            w_end = cw - im.shape[1] - w_beg
            im = np.pad(im, [(h_beg, h_end), (w_beg, w_end)] + [(0, 0)] * (im_rank - 2), mode='edge')
        if ch != im.shape[0] or cw != im.shape[1]:
            h_beg = (im.shape[0] - ch) // 2
            h_end = im.shape[0] - ch - h_beg
            w_beg = (im.shape[1] - cw) // 2
            w_end = im.shape[1] - cw - w_beg
            im = im[h_beg:-h_end, w_beg:-w_end]
        return im

    def process_data(self, index, input_list, output_list):

        im = input_list[index]
        im_rank = len(im.shape)

        # resize shorter edge (can have randomness)
        if self.shorter_edge_length is not None:
            if isinstance(self.shorter_edge_length, list):
                # random generation
                n = len(self.shorter_edge_length)
                if n == 1:
                    k = 0
                else:
                    k = np.random.randint(low=0, high=n)
                target_shorter_edge = self.shorter_edge_length[k]
            else:
                target_shorter_edge = self.shorter_edge_length

            actual_shorter_edge = min(im.shape[0], im.shape[1])
            if actual_shorter_edge != target_shorter_edge:
                zoom_factor = [target_shorter_edge / actual_shorter_edge] * 2
                zoom_factor.extend([1]*(im_rank-2))
                im = scipy.ndimage.zoom(im, zoom_factor)

        # crop center, add pad if necessary
        canvas_size = None
        if self.center_patch_size is None:
            if not self.use_random and self.crop_size is not None:
                canvas_size = self.crop_size
        else:
            if self.crop_size is None:
                canvas_size = self.center_patch_size
            else:
                if self.use_random:
                    canvas_size = [
                        max(self.center_patch_size[0], self.crop_size[0]),
                        max(self.center_patch_size[1], self.crop_size[1])
                    ]
                else:
                    canvas_size = self.crop_size

        if canvas_size is not None:
            im = self.robust_center_crop(im, canvas_size)

        # crop final patch (can have randomness)
        if self.crop_size is not None:
            canvas_size = im.shape[0:2]
            if self.crop_size != canvas_size:
                assert self.use_random, "internal error: should not reach here is not using random"
                h_beg = np.random.randint(low=0, high=canvas_size[0]-self.crop_size[0])
                w_beg = np.random.randint(low=0, high=canvas_size[1]-self.crop_size[1])
                im = im[h_beg:(h_beg+self.crop_size[0]), w_beg:(w_beg+self.crop_size[1])]

        if self.image_color_scaling is not None:
            im = im * self.image_color_scaling + (1-self.image_color_scaling) * 0.5

        output_list[index] = im

    def next_batch(self, batch_size):

        # get raw data
        data_list = self.data_module(batch_size)
        raw_data = data_list[self._data_field_index]

        if isinstance(raw_data, np.ndarray):
            n = raw_data.shape[0]
        else:
            n = len(raw_data)

        processed_data = [None]*n
        for i in range(n):  # implement in an easy to parallel way
            self.process_data(i, raw_data, processed_data)

        processed_data = np.concatenate([np.expand_dims(v, axis=0) for v in processed_data], axis=0)

        output_data_list = copy(data_list)
        output_data_list[self._data_field_index] = processed_data

        return output_data_list

