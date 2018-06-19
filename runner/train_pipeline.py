import zutils.tf_graph_utils as tgu
from zutils.py_utils import *

from runner.base_pipeline import Pipeline as BasePipeline

from runner.train_pipeline_traindef import TrainDef as PipelineTrainDef
from runner.train_pipeline_traindef import TrainDefOptions as PipelineTrainDefOptions

from runner.data_op import tf_variable_from_data_module

import runner.preprocessing_data_module_wrapper

TrainDefOptions = PipelineTrainDefOptions


class Pipeline(BasePipeline):

    TrainDef = PipelineTrainDef
    TrainDefOptions = PipelineTrainDefOptions

    def __init__(self, model_dir=None, **kwarg):

        super().__init__(model_dir, **kwarg)

        # gpu names
        self.ps_device_name, self.gpu_names = self._init_gpu_names()

    @staticmethod
    def _init_gpu_names(num_gpus=None, ps_cpu=True):
        # check devices
        ps_device = None
        gpu_names = None
        all_gpu_names = tgu.get_available_gpus()
        assert len(all_gpu_names)<=1, \
            "ERROR: this code does not support multiple devices. Use CUDA_VISIBLE_DEVICES=... to specify the GPU."

        return ps_device, gpu_names

    def init_logger_saver(self):

        self.init_logger()

        with self.graph.as_default():

            print("* Link logger with trainer")
            for train_obj in self.train.groups.values():
                train_obj.trainer.set_logger(self.logger)

            # saver
            print("* Define net+trainer saver")
            saver_kwargs = dict()
            if self.opt.keep_checkpoint_every_n_hours is not None:
                saver_kwargs["keep_checkpoint_every_n_hours"] = self.opt.keep_checkpoint_every_n_hours
            if self.opt.max_checkpoint_to_keep is not None:
                saver_kwargs["max_to_keep"] = self.opt.max_checkpoint_to_keep
            self.saver = tgu.MultiDeviceSaver(var_list=self.train.saveable_variables, **saver_kwargs)
            print("* Define net saver")
            self.net_saver = tgu.MultiDeviceSaver(var_list=self.train.net_variables, **saver_kwargs)

    def data_module_preprocess(self, raw_data_module, mode=None):
        return runner.preprocessing_data_module_wrapper.Net(raw_data_module, self.opt_dict, mode)

    def data_module_to_tensor(self, raw_data_module, data_fields, batch_size):
        preprocessed_data_module = self.data_module_preprocess(raw_data_module, mode="deterministic")
        return tf_variable_from_data_module(preprocessed_data_module, batch_size, data_fields)

    def output_scaled_color_image(self, im):
        if self.opt.image_color_scaling:
            im = (im - (1-self.opt.image_color_scaling)*0.5) / self.opt.image_color_scaling
        return im
