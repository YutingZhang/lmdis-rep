import net_modules.gen
import numpy as np
import tensorflow as tf
from net_modules.common import default_activation

import math

import zutils.tf_math_funcs as tmf
from model.model import Factory
from model.options import PipelineOptionDefinition
from runner.train_pipeline import Pipeline as BasePipeline
from zutils.option_struct import OptionDef
from zutils.py_utils import *
import zutils.tf_graph_utils as tgu
from runner.resumable_data_module_wrapper import Net as ResumableDataModuleWrapper

net_factory = net_modules.gen.get_net_factory
net_instance = net_modules.gen.get_net_instance


class PipelineNetDef(BasePipeline):   # base class for define the pipeline

    def __init__(self, pipeline_scope_name=None, user_options=None, *args, **kwargs):

        # initialize base pipeline object
        super().__init__(*args, **kwargs)

        # get pipeline options
        if user_options is None:
            user_options = dict()
        self.option_def = OptionDef(user_options, PipelineOptionDefinition)
        opt_struct = self.option_def["all"]
        self.set_options(opt_struct)

        # set pipeline name
        if pipeline_scope_name is None:
            pipeline_scope_name = "vae"
            # remark: historically used "vae", and keep it for the compatiblity to the pretrained models
        self.scope_name = pipeline_scope_name

        # init all models ------------------------------------------------------------------
        with self.graph.as_default():
            # get model factory (do it first, otherwise cannot create others)
            self.net_factory = Factory(self.scope_name, self.opt)

        the_default_nonlinearity = self.net_factory.get_non_linearity()
        with self.graph.as_default(), default_activation(the_default_nonlinearity):

            self.train = NetDefTrain(self)
            self.init_logger_saver()

            # build posterior param
            self.posterior = NetDefPosterior(self)

            print("== Pipeline initialization is done")

    def load_data_module(self, subset_name, mod2op_func, extra_fields=None, extra_options=None):

        if extra_options is None:
            extra_options = dict()

        batch_size = None
        if isinstance(mod2op_func, (int, float)):
            batch_size = copy(mod2op_func)

            def standard_mod2op_func(dm, df):
                return self.data_module_to_tensor(dm, df, batch_size)

            mod2op_func = standard_mod2op_func

        if extra_fields is None:
            extra_fields = []
        elif isinstance(extra_fields, (tuple, set)):
            extra_fields = list(extra_fields)
        elif isinstance(extra_fields, str):
            extra_fields = [extra_fields]

        raw_data_module = self.net_factory.data_net_module(subset_name, extra_options=extra_options)

        data_fields = ["data"]
        data_fields.extend(self.opt.condition_list)

        for k in extra_fields:
            if k not in data_fields and k in raw_data_module.output_keys():
                data_fields.append(k)

        # ------------------
        i = 0
        data_key2ind = dict()
        for v in raw_data_module.output_keys():
            data_key2ind[v] = i
            i += 1
        # ------------------

        is_train = False
        if "is_train" in extra_options:
            is_train = extra_options["is_train"]
            del extra_options["is_train"]

        data_list = mod2op_func(raw_data_module, data_fields)
        if not isinstance(data_list, (list, tuple)):
            data_list = [data_list]
        condition_list = []

        batch_size_from_data = tmf.get_shape(data_list[0])[0]
        if batch_size is None:
            batch_size = batch_size_from_data
        else:
            assert batch_size == batch_size_from_data, \
                "inconsistency between data batch_size and specifiy batch_size"

        if not is_train and self.opt.rotate_batch_samples:
            raw_data_module = ResumableDataModuleWrapper(
                raw_data_module, _num_sample_factors=batch_size
            )
            data_list = mod2op_func(raw_data_module, data_fields)
    
            data_list = tgu.batch_rotating_data_buffer(
                data_list, batch_size=batch_size, capacity=batch_size*2, enqueue_many=True
            )
            data_fields.append("boolmask_for_dominating_sample_in_batch_rotating")
            condition_list.append("boolmask_for_dominating_sample_in_batch_rotating")

        if not isinstance(data_list, (list, tuple)):
            data_list = [data_list]

        data = OrderedDict()
        for k, v in zip(data_fields, data_list):
            data[k] = v

        # prepare condition dict
        condition_list.extend(self.opt.condition_list)
        cond_info_dict = dict()

        for cond_name in condition_list:

            cond_prefix, cond_type, cond_postfix = tgu.name_parts(cond_name)

            assert cond_name in data, "not such data field for condition"
            cond_data = data[cond_name]
            cond_dict = dict()

            if cond_type == "class":
                # class condition
                cond_dict["indexes"] = cond_data
                cond_dict["num"] = raw_data_module.output_ranges()[data_key2ind["class"]]
            elif cond_type == "landmark":       # y, x
                # landmark condition
                cond_dict["size"] = tmf.get_shape(data["data"])[1:3]
                # convert to normalized coordinate
                if tmf.get_shape(cond_data)[2] > 2:
                    cond_dict["gate"] = cond_data[:, :, 0]
                    cond_data = cond_data[1:]
                else:
                    cond_dict["gate"] = tf.ones_like(cond_data[:, :, 0])

                cond_dict["location"] = \
                    (tf.to_float(cond_data)+0.5) / math.sqrt(cond_dict["size"][0]*cond_dict["size"][1])

                assert self.opt.image_crop_size is None, "landmark condition does not work with image_crop_size"

            elif cond_type == "optical_flow":   # y, x
                h, w = tmf.get_shape(cond_data)[1:3]
                cond_data = tf.to_float(cond_data)
                cond_dict["flow"] = cond_data / math.sqrt(h*w)
                flow_offset_name = os.path.join(cond_prefix, "optical_flow_offset" + cond_postfix)
                cond_dict["offset"] = data[flow_offset_name]

                assert self.opt.image_crop_size is None, "optical_flow condition does not work with image_crop_size"

            else:
                cond_dict = cond_data   # just a tensor, not dict
            assert rbool(cond_dict), "internal error: cond_dict is not set"

            cond_info_dict[cond_name] = cond_dict

        return raw_data_module, data, cond_info_dict

    def prefix_data_module(self, data_module):
        if self.opt.data_class_list is not None:
            assert hasattr(data_module, "limit_to_classes"), \
                "Data loader do not support limit_to_classes"
            data_module.limit_to_classes(self.opt.data_class_list)


class NetDefTrain(BasePipeline.TrainDef):

    def __init__(self, pipeline):

        super().__init__(pipeline, pipeline.opt)

        # prepare training data
        print("* Define training model")
        print(" - data module")

        self.raw_data_module, self.input, self.cond_info_dict = pipeline.load_data_module(
            pipeline.opt.train_subset, self.create_data_tensor,
            extra_options={
                "shuffle": self.train_use_shuffle(),
                "is_train": True,
            }
        )
        pipeline.prefix_data_module(self.raw_data_module)

        # Parallel condition
        flat_cond, wrap_cond_func = recursive_flatten_with_wrap_func(tmf.is_tf_data, self.cond_info_dict)
        flat_cond_len = len(flat_cond)
        # REMARK: ****************** currently, cannot learn/finetune condition embedding

        # create data input
        stacked_data_input = list()
        stacked_data_input.append(self.input["data"])
        stacked_data_input.extend(flat_cond)

        # expand data input to kwargs
        def train_data_kwargs(data_tensors):
            tp = 0
            kwa = dict()
            kwa["data_tensor"] = data_tensors[tp]
            tp += 1
            kwa["cond_info_dict"] = wrap_cond_func(data_tensors[tp:tp+flat_cond_len])
            tp += flat_cond_len
            return kwa

        # --- training net definition

        def standard_ae_training_net(data_tensors, default_reuse=None):
            loss, disp_outputs, full_collection, _ = \
                pipeline.net_factory.ae_training_net(
                    default_reuse=default_reuse,
                    **train_data_kwargs(data_tensors))
            dvout = OrderedDict(graph=full_collection)
            return loss, disp_outputs, dvout

        ae_out = self.get_train_output(stacked_data_input, standard_ae_training_net)

        ae_group = self.create_group("ae")

        ae_group.loss = ae_out.loss
        ae_group.outputs = ae_out.display_outputs
        ae_group.ps_outputs = ae_out.ps_device_outputs
        ae_group.device_outputs = ae_out.device_outputs
        ae_group.data_module = self.data_module


        # handle extra outputs --------------------
        self.init_extra_outputs_and_indicators()
        self.trainer_init(scope=pipeline.scope_name + "/trainer")


class NetDefPosterior:

    def __init__(self, pipeline):
        print("* Define posterior param model")

        self.data_module, posterior_input, self.cond_info_dict = pipeline.load_data_module(
            pipeline.opt.test_subset, pipeline.opt.test_batch_size, extra_fields=["class"])
        pipeline.prefix_data_module(self.data_module)

        if "class" in posterior_input:
            posterior_label = posterior_input["class"]
        else:
            posterior_label = tf.zeros([posterior_input["data"].get_shape()[0], 1], dtype=tf.int32)

        self.input = posterior_input
        posterior_output, posterior_aux_out = pipeline.net_factory.posterior_net(self.input["data"], cond_info_dict=self.cond_info_dict)
        self.outputs = posterior_aux_out
        self.outputs["posterior_param"] = posterior_output
        self.outputs["class_label"] = posterior_label

