import prettytensor as pt
import tensorflow as tf

import net_modules
import zutils.tf_graph_utils as tgu
import zutils.tf_math_funcs as tmf
from zutils.py_utils import *

net_factory = net_modules.gen.get_net_factory
try_net_factory = net_modules.gen.try_get_net_factory
net_instance = net_modules.gen.get_net_instance
try_net_instance = net_modules.gen.try_get_net_instance
class_instance = net_modules.gen.get_class_instance

epsilon = tmf.epsilon
epsilon2 = tmf.epsilon2


class Factory:

    # Initialization ----------------------------------------------------------------------------

    def __init__(self, model_name, model_option):
        self._name = model_name
        if hasattr(model_option, "get_namedtuple"):
            self.opt = model_option.get_namedtuple()
        else:
            self.opt = model_option
        self._recon_factory_cache = None

    def get_non_linearity(self):
        if self.opt.non_linearity == "elu":
            return tmf.elu
        elif self.opt.non_linearity == "relu":
            return tf.nn.relu
        elif self.opt.non_linearity == "leaky_relu":
            return tmf.leaky_relu
        else:
            raise ValueError("Unrecognized non_linearity")


    # Data ----------------------------------------------------------------------------

    def data_net_module(self, subset_name, extra_options):

        # when cannot get a factory, get an instance and wrap it as an op
        extra_options = copy(extra_options)
        if extra_options is None:
            extra_options = dict()
        is_train = False
        if "is_train" in extra_options:
            is_train = extra_options["is_train"]
            del extra_options["is_train"]

        if is_train:
            data_name = self.opt.data_name
            data_options = self.opt.data_options
        else:
            data_name = self.opt.test_data_name
            data_options = self.opt.test_data_options

        if data_options is None:
            data_options = dict()
        data_options = {**extra_options, **data_options}
        # data_mod = net_instance("data", self.opt.data_name, subset_name)
        data_mod = call_func_with_ignored_args(
            net_instance, "data", data_name, subset_name, options=data_options
        )

        return data_mod

    # condition -------------------------------------------------------------------------
    def condition_subnet(self, info_dict):
        """
        Parse the condition input into condition_tensor
        Remark: conditions mean all the inputs besides the data
        """
        if info_dict is None:
            return None

        all_conditions = []
        for k, v in info_dict.items():
            if not isinstance(v, dict):
                v = {"value": v}
            bk = tgu.bare_name(k)
            all_conditions.append({"type": bk, "name": k, **v})

        if not all_conditions:
            return None
        return all_conditions

    # Encoder ---------------------------------------------------

    def _encoding_subnet(
            self, encoder_name, data_tensor, output_channel, condition_tensor=None, options=None,
            extra_inputs=None
    ):
        enc_factory = call_func_with_ignored_args(
            net_factory, "encoder", encoder_name,
            output_channels=output_channel, options=options)
        latent_tensor = call_func_with_ignored_args(
            enc_factory, data_tensor, condition_tensor=condition_tensor, extra_inputs=extra_inputs)

        default_extra_outputs = dict()
        default_extra_outputs["save"] = dict()
        default_extra_outputs["extra_recon"] = dict()
        default_extra_outputs["cond"] = dict()
        default_extra_outputs["for_decoder"] = dict()
        if isinstance(latent_tensor, (tuple, list)):
            extra_outputs = latent_tensor[1]
            latent_tensor = latent_tensor[0]
        else:
            extra_outputs = dict()

        extra_outputs = {**default_extra_outputs, **extra_outputs}

        return latent_tensor, extra_outputs

    def encoding_subnet(self, data_tensor, condition_tensor=None, extra_inputs=None, encoding_scope=None):
        if encoding_scope is None:
            encoding_scope = "encoder"
        with tf.variable_scope(encoding_scope):
            latent_tensor, extra_outputs = self._encoding_subnet(
                self.opt.encoder_name, data_tensor, self.opt.latent_dim,
                condition_tensor=condition_tensor, options=self.opt.encoder_options,
                extra_inputs=extra_inputs
            )
        return latent_tensor, extra_outputs

    # Decoder ---------------------------------------------------

    def _decoding_subnet(self, samples, condition_tensor=None, options=None, extra_inputs=None):

        if condition_tensor is not None:
            batch_size = tmf.get_shape(samples)[0]
            def rep_to_batch_size(x):
                x_batch_size = tmf.get_shape(x)[0]
                if batch_size == x_batch_size:
                    return x
                rep_factor = batch_size // x_batch_size
                assert rep_factor == batch_size / x_batch_size, "sample num factor is not integer"
                return tmf.rep_sample(x, rep_factor)
            condition_tensor = recursive_apply(
                tmf.is_tf_data,
                rep_to_batch_size,
                condition_tensor
            )

        output_param_num = self._recon_factory().param_num()
        factory_arg_prefix = ["decoder"]
        if condition_tensor is not None:
            if hasattr(self.opt, "condition_at_latent") and self.opt.condition_at_latent is not None:
                factory_arg_prefix = [
                    "cond_decoder",
                    "generic_" + self.opt.condition_at_latent + "_at_begin"
                ]

        dec_factory = call_func_with_ignored_args(
            net_factory,
            *factory_arg_prefix, self.opt.decoder_name, output_param_num, options=options
        )

        reconstructed = call_func_with_ignored_args(
            dec_factory, samples, condition_tensor=condition_tensor, extra_inputs=extra_inputs)

        default_extra_outputs = dict()
        default_extra_outputs["save"] = dict()
        default_extra_outputs["extra_recon"] = dict()
        default_extra_outputs["cond"] = dict()

        if isinstance(reconstructed, (tuple, list)):
            extra_outputs = reconstructed[1]
            reconstructed = reconstructed[0]
        else:
            extra_outputs = dict()
        extra_outputs = {**default_extra_outputs, **extra_outputs}

        return reconstructed, extra_outputs

    def decoding_subnet(self, samples, condition_tensor=None, extra_inputs=None):
        with tf.variable_scope("decoder"):
            return self._decoding_subnet(
                samples, condition_tensor=condition_tensor, options=self.opt.decoder_options,
                extra_inputs=extra_inputs
            )

    # Objectives ---------------------------------------------------

    def _get_recon_factory(self, recon_name, *args, **kwargs):
        fct = try_net_factory("recon", self.opt.recon_name, *args, **kwargs)
        if fct is None:
            fct = call_func_with_ignored_args(
                try_net_factory,
                "recon", "generic_single_dist",
                recon_name, *args, **kwargs)
        assert fct is not None, "Cannot get recon factory"
        return fct

    def _recon_factory(self, *args, **kwargs):
        if self._recon_factory_cache is None:
            self._recon_factory_cache = self._get_recon_factory(
                self.opt.recon_name, *args, options=self.opt.recon_options, **kwargs
            )
        return self._recon_factory_cache

    def recon_subnet(self, reconstructed, data_tensor):
        rep_num = tmf.get_shape(reconstructed)[0] // tmf.get_shape(data_tensor)[0]
        data_tensor = tmf.rep_sample(data_tensor, rep_num)
        with tf.variable_scope("recon"):
            elt_recon_loss = \
                self._recon_factory()(reconstructed, data_tensor)
        return elt_recon_loss

    def decoded_vis_subnet(self, reconstructed):
        with tf.variable_scope("recon"):
            vis_aug = self._recon_factory().visualize(reconstructed)
        return vis_aug

    def recon_and_nll_subnet(self, samples, data_tensor, enc_extra_outputs=None, **kwargs):
        sample_num = tmf.get_shape(samples)[0]//tmf.get_shape(data_tensor)[0]

        dec_extra_kwargs = dict()
        dec_extra_kwargs["extra_inputs"] = dict()
        if enc_extra_outputs is not None and "for_decoder" in enc_extra_outputs:
            dec_extra_inputs = enc_extra_outputs["for_decoder"]
            dec_extra_kwargs["extra_inputs"] = dec_extra_inputs
        dec_extra_kwargs["extra_inputs"]["data"] = data_tensor

        reconstructed, dec_extra_outputs = self.decoding_subnet(samples, **dec_extra_kwargs, **kwargs)

        if "recon_batch_size" in dec_extra_outputs and dec_extra_outputs["recon_batch_size"] is not None:
            recon_batch_size = dec_extra_outputs["recon_batch_size"]
        else:
            recon_batch_size = tmf.get_shape(reconstructed)[0]

        elt_recon_nll = self.recon_subnet(reconstructed[:recon_batch_size], tmf.rep_sample(data_tensor, sample_num))
        recon_nll = tmf.sum_per_sample(elt_recon_nll)

        total_recon_nll = recon_nll

        return reconstructed, total_recon_nll, recon_nll, dec_extra_outputs

    def ae_training_net(self, data_tensor, default_reuse=None, cond_info_dict=None):

        sub_collections = tgu.SubgraphCollectionSnapshots()
        sub_collections.sub_snapshot("_old")

        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope(self._name, reuse=default_reuse) as scope:
                condition_tensor = self.condition_subnet(cond_info_dict)
        # remark: condition subnet not trainable
        sub_collections.sub_snapshot("cond")

        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope(self._name, reuse=default_reuse) as scope:
                latent_param, enc_extra_outputs = self.encoding_subnet(
                    data_tensor, condition_tensor=condition_tensor)

                decoder_collections = tgu.SubgraphCollectionSnapshots()
                decoder_collections.sub_snapshot("_old")
                reconstructed_raw, recon_nll,  _, _ = self.recon_and_nll_subnet(
                    latent_param, data_tensor,
                    enc_extra_outputs=enc_extra_outputs,
                    condition_tensor=condition_tensor
                )
                recon_loss = tf.reduce_mean(recon_nll)
                reconstructed, _ = self.decoded_vis_subnet(reconstructed_raw)
                decoder_collections.sub_snapshot("decoder")

        sub_collections.sub_snapshot("full")
        full_collection = sub_collections["full"]

        extra_output_dict = dict(
            cond_graph=sub_collections["cond"]
        )

        recon_loss *= self.opt.recon_weight
        loss = recon_loss

        loss_breakdown = OrderedDict()
        loss_breakdown["recon"] = recon_loss

        return loss, loss_breakdown, full_collection, extra_output_dict

    # Posterior ------------------------------------------------------------------------------------

    def posterior_net(self, data_tensor, cond_info_dict):

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope(self._name, reuse=True) as scope:
                condition_tensor = self.condition_subnet(cond_info_dict)
                enc_outputs, enc_extra_outputs = self.encoding_subnet(
                    data_tensor, condition_tensor=condition_tensor
                )
                reconstructed, dec_extra_outputs = self.decoding_subnet(
                    enc_outputs, condition_tensor=condition_tensor,
                    extra_inputs=enc_extra_outputs['for_decoder']
                )
                decoded_vis, decoded_param_tensor = self.decoded_vis_subnet(reconstructed)

        decoded_out = dict()
        decoded_out["vis"] = decoded_vis
        decoded_out["param"] = decoded_param_tensor
        aux_out = dict()
        aux_out["decoded"] = decoded_out    # extracts "save" field in self.reconstructed_base_aux_out
        aux_out["encoded"] = enc_extra_outputs["save"]
        aux_out["data"] = data_tensor

        latent_param_dict = dict(
            value=enc_outputs
        )

        return latent_param_dict, aux_out

