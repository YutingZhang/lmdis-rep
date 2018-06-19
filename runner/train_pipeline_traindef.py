import tensorflow as tf
import zutils.tf_graph_utils as tgu

from zutils.py_utils import *

from collections import OrderedDict

from copy import copy

import functools

from runner.network_trainer import NeuralNetworkTrainer
from runner.network_trainer import single_device_training_net

import runner.resumable_data_module_wrapper

import zutils.tf_math_funcs as tmf

from runner.data_op import tf_variable_from_data_module


class TrainDefOptions:

    @staticmethod
    def trainer(p):

        # batch size
        p["train_batch_size"] = 256

        # data shuffling
        # None: no shuffling
        # > 1: number with respect to batch size
        # "epoch": shuffle the whole dataset
        p["train_shuffle_capacity"] = None

        p["train_color_jittering"] = False
        p["train_random_mirroring"] = False

        # optimization
        p["weight_decay"] = 5e-4
        p["optimizer"] = "Adam"
        p["optimizer_options"] = dict()
        p["learning_rate"] = 1e-3
        p["max_epochs"] = 500

        # monitoring
        p["disp_time_interval"] = 2
        p["snapshot_interval"] = 3600
        p["test_steps"] = None
        p["permanent_snapshot_step_list"] = []
        p["snapshot_step_list"] = []
        p["keep_checkpoint_every_n_hours"] = None
        p["max_checkpoint_to_keep"] = None

        # batch normalization
        p["bn_infer_steps"] = 500
        p["bn_mimimum_steps_to_infer"] = 0

        # image preprocessing
        p["image_shorter_edge_length"] = None
        if p["image_shorter_edge_length"] is not None:
            p["image_shorter_edge_length_list"] = None
        p["image_center_patch_size"] = None
        p["image_crop_size"] = None
        p["image_color_scaling"] = None


class TrainDef:

    def __init__(self, pipeline, options=None):

        # set up options
        if options is None:
            options = pipeline.opt
        assert options is not None, "options must be specified"
        if isinstance(options, dict):
            options = dict2namedtuple(options, "option_tuple")
        self.opt = options

        # batch_size
        self._data_module = None
        self._data_tensor = None
        self._batch_size = None
        self._actual_batch_size = None
        self._called_create_data_tensor = False
        self._called_trainer_init = False
        self._called_set_batch_size = False

        # init
        self.pipeline = pipeline
        self.net_variables = None
        self.groups = OrderedDict()
        self.train_func_raw_outputs = None
        self.trainer_variables = None
        self.saveable_variables = None

        # define train group class
        class TrainGroup:
            def __init__(self):
                self.data_module = None
                self.disp_prefix = None
                self.loss = None
                self.outputs = OrderedDict()
                self.extra_outputs = None
                self.extra_output_indicators = None
                self.selected_extra_outputs = None
                self.minimizer_kwargs = OrderedDict()  # used for minimize
                self.update_ops = None
                self.aux_loss = None
                self.ps_outputs = None      # must have "graph" field, for the subgraph collection
                self.device_outputs = None  # must have "graph" field, for the subgraph collection
                self.trainer = None
                self.weight_decay = options.weight_decay
                self.max_epochs = options.max_epochs
                self.optimizer = options.optimizer
                self.optimizer_options = options.optimizer_options  # used for optimizer
                self.disp_time_interval = options.disp_time_interval
                self.learning_rate = options.learning_rate

        self.TrainGroup = TrainGroup

    def set_batch_size(self, batch_size=None):
        assert not self._called_set_batch_size, \
            "batch_size should not be set more than once"
        self._called_set_batch_size = True

        self._batch_size = batch_size

        basic_batch_size = None
        if self._batch_size is None:
            if self.opt is not None:
                basic_batch_size = self.opt.train_batch_size
        else:
            basic_batch_size = self._batch_size

        assert basic_batch_size is not None, "train batch size is not specified"

        self._actual_batch_size = basic_batch_size  # single device setting

    @property
    def batch_size(self):
        if self._actual_batch_size is None:
            self.set_batch_size()
        return self._actual_batch_size

    @property
    def data_module(self):
        return self._data_module

    @property
    def data_tensor(self):
        return self._data_tensor

    def train_use_shuffle(self):
        if self.opt.train_shuffle_capacity is not None:
            if self.opt.train_shuffle_capacity == "full":
                return True
            if self.opt.train_shuffle_capacity == "on":
                return True
            if self.opt.train_shuffle_capacity:
                return True
        return False

    def create_data_tensor(self, raw_data_module, output_index=0):
        # set up: self.data_module and self.data_tensor
        assert not self._called_create_data_tensor, \
            "data_tensor should not be create twice"
        self._called_create_data_tensor = True

        preprocessed_data_module = self.pipeline.data_module_preprocess(raw_data_module, mode="random")
        self._data_module = runner.resumable_data_module_wrapper.Net(preprocessed_data_module)
        unshuffled_data_tensor = tf_variable_from_data_module(
            self.data_module, self.batch_size, output_index)

        tsc = self.opt.train_shuffle_capacity
        maximum_shuffle_capacity = None
        if self.train_use_shuffle():
            if tsc == "full":
                maximum_shuffle_capacity = self.data_module.num_samples()
            elif tsc == "on":
                # use shuffle provided by the data loader
                pass
            elif tsc >= 1:
                maximum_shuffle_capacity = tsc * self.batch_size
                if maximum_shuffle_capacity > self.data_module.num_samples():
                    maximum_shuffle_capacity = self.data_module.num_samples()

        if maximum_shuffle_capacity is not None:
            minimum_shuffle_capacity = 3 * self.batch_size
            if tmf.is_tf_data(unshuffled_data_tensor):
                unshuffled_data_tensor = [unshuffled_data_tensor]
            data_tensor = tf.train.shuffle_batch(
                unshuffled_data_tensor, batch_size=self.batch_size,
                capacity=maximum_shuffle_capacity,
                min_after_dequeue=minimum_shuffle_capacity,
                enqueue_many=True)
        else:
            data_tensor = tgu.sequential_data_buffer(
                unshuffled_data_tensor, batch_size=self.batch_size, capacity=self.batch_size*3,
                enqueue_many=True
            )

        if self.opt.train_color_jittering or self.opt.train_random_mirroring:
            if isinstance(data_tensor, list):
                image = data_tensor[0]
            else:
                image = data_tensor

            if self.opt.train_color_jittering:
                if self.opt.image_color_scaling is not None:
                    clip_value_min = (1 - self.opt.image_color_scaling) * 0.5
                    image = (image - clip_value_min) / self.opt.image_color_scaling
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                # image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
                if self.opt.image_color_scaling is not None:
                    clip_value_min = (1 - self.opt.image_color_scaling) * 0.5
                    image = image * self.opt.image_color_scaling + clip_value_min

            if self.opt.train_random_mirroring:
                image_flip = tf.reverse(image, axis=[2])
                im_inp_shape = tmf.get_shape(image)
                flip_ind = tf.random_uniform(
                    [im_inp_shape[0]]+[1]*(len(im_inp_shape)-1), minval=0., maxval=1., dtype=tf.float32) > 0.5
                image = tf.where(tf.tile(flip_ind, [1]+im_inp_shape[1:]), image_flip, image)

            if isinstance(data_tensor, list):
                data_tensor[0] = image
            else:
                data_tensor = image

        self._data_tensor = data_tensor

        return data_tensor

    def create_group(self, key):
        assert not self._called_trainer_init, \
            "group cannot be created after trainer_init"

        g = self.TrainGroup()
        self.groups[key] = g
        return g

    def get_group(self, key):
        return self.groups[key]

    def get_train_output(self, inputs, standard_training_net):
        train_out, self.net_variables = single_device_training_net(inputs, standard_training_net)
        self.train_func_raw_outputs = train_out
        return train_out

    def _get_extra_output_indicators(self, extra_outputs):
        the_extra_outputs = extra_outputs
        the_indicators = recursive_indicators(tmf.is_tf_data, the_extra_outputs)
        return the_indicators

    def _select_extra_outputs(self, extra_outputs, the_indicators):
        the_selected = recursive_select(extra_outputs, the_indicators)
        return the_selected

    def init_extra_outputs_and_indicators(self):

        def remove_common_fields(dc):
            return OrderedDict(filter(
                lambda elt: elt[0] not in ("graph", "cond_graph") and elt[1] is not None,
                dc.items()
            ))

        for v in self.groups.values():
            v.extra_outputs = remove_common_fields(v.ps_outputs)
            v.extra_output_indicators = self._get_extra_output_indicators(v.extra_outputs)

    def trainer_init(self, scope=None):
        assert not self._called_trainer_init, \
            "trainer can be init only once"

        generic_trainer_kwargs = dict()
        if self.opt.test_steps is not None:
            generic_trainer_kwargs["test_steps"] = self.opt.test_steps
            generic_trainer_kwargs["test_func"] = self.pipeline._snapshot_test

        is_first_group = True
        first_trainer = None
        trainer_variables = []
        for k, v in self.groups.items():
            print(" - net trainer for %s" % k)
            # complete arguments
            if "var_list" not in v.minimizer_kwargs:
                v.minimizer_kwargs["var_list"] = \
                    v.ps_outputs["graph"].get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            if v.update_ops is None:
                v.update_ops = list()
                for dev_out in v.device_outputs.values():
                    v.update_ops.extend(dev_out["graph"].get_collection(tf.GraphKeys.UPDATE_OPS))
            v.full_var_list = v.ps_outputs["graph"].get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # weight decay
            print("  - weight decay:")
            scaled_weight_decay, decayed_vars = tgu.regularization_sum(
                trainable_variables=v.minimizer_kwargs["var_list"],
                default_scale=v.weight_decay,
                display=True)
            loss_variable = v.loss
            if decayed_vars:
                v.outputs["weight_decay"] = scaled_weight_decay
                loss_variable += scaled_weight_decay
            else:
                print("   - No weight decay applied")

            extra_kwargs = copy(generic_trainer_kwargs)
            if is_first_group:
                # define snapshot
                snapshot_interval = self.opt.snapshot_interval
                extra_kwargs["snapshot_func"] = self.pipeline.periodic_snapshot_func
                if self.pipeline.periodic_snapshot_func is None:
                    snapshot_interval = None
                extra_kwargs["snapshot_interval"] = snapshot_interval
                extra_kwargs["permanent_snapshot_step_list"] = self.opt.permanent_snapshot_step_list
                extra_kwargs["snapshot_step_list"] = self.opt.snapshot_step_list

            else:
                # share snapshot
                extra_kwargs["snapshot_sharing"] = first_trainer

            if v.aux_loss is None:
                all_aux_loss = OrderedDict()
                all_aux_loss_elt = OrderedDict()
                for dk, ds in v.device_outputs.items():
                    all_aux_loss_k = ds["graph"].get_collection("aux_loss")
                    all_aux_loss_elt_dk = OrderedDict()
                    if all_aux_loss_k:
                        all_aux_loss[dk] = functools.reduce(
                            lambda a1, a2: a1+a2, all_aux_loss_k)

                        ti = 0
                        for t in all_aux_loss_k:
                            ti = ti+1
                            if hasattr(t, 'disp_name'):
                                if t.disp_name in all_aux_loss_elt_dk:
                                    all_aux_loss_elt_dk[t.disp_name] += t
                                else:
                                    all_aux_loss_elt_dk[t.disp_name] = t
                    all_aux_loss_elt[dk] = all_aux_loss_elt_dk

                all_aux_loss = OrderedDict(sorted(all_aux_loss.items(), key=lambda t: t[0]))
                all_aux_loss_elt = OrderedDict(sorted(all_aux_loss_elt.items(), key=lambda t: t[0]))

                if all_aux_loss:
                    v.aux_loss = all_aux_loss['ps']
                    v.aux_loss_elt = all_aux_loss_elt['ps']

            loss_variable_full = loss_variable
            disp_variable_dict = v.outputs
            if v.aux_loss is not None:
                loss_variable_full += v.aux_loss
                disp_variable_dict["aux_loss"] = v.aux_loss
                for aux_k, aux_v in v.aux_loss_elt.items():
                    disp_variable_dict["aux_" + aux_k] = aux_v

            loss_variable.disp_name = "[MAIN_LOSS]"

            v.full_loss = loss_variable_full

            if v.selected_extra_outputs is None:
                v.selected_extra_outputs = self._select_extra_outputs(
                    v.extra_outputs, v.extra_output_indicators
                )

            v.trainer = NeuralNetworkTrainer(
                data_module=v.data_module,
                loss_tensor=loss_variable_full,
                solver_type=v.optimizer,
                solver_kwargs=v.optimizer_options,
                disp_tensor_dict=v.outputs,
                minimizer_kwargs=v.minimizer_kwargs,
                update_ops=v.update_ops,
                max_epochs=v.max_epochs,
                disp_time_interval=v.disp_time_interval,
                disp_prefix=v.disp_prefix,
                learning_rate=v.learning_rate,
                scope=scope,
                extra_output_tensors=v.selected_extra_outputs,
                **extra_kwargs
            )

            v.trainer.setup_avgvar_update(
                full_var_list=v.full_var_list,
                forward_steps=self.opt.bn_infer_steps,
                minimum_update_steps=self.opt.bn_mimimum_steps_to_infer,
            )

            if is_first_group:
                first_trainer = v.trainer

            trainer_variables += v.trainer.saveable_variables

            is_first_group = False

        trainer_variables.append(self.pipeline.global_step)
        self.trainer_variables = list(set(trainer_variables))

        self.saveable_variables = list(set(self.net_variables + self.trainer_variables))

