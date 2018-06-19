from runner.train_pipeline_traindef import TrainDefOptions


# default options of the model
class ModelOptionDefinition:

    @staticmethod
    def main_model(p):

        # non-linearity
        p["non_linearity"] = "leaky_relu"

        # dataset name
        p["data_name"] = "mnist_binary_rbm"
        p["data_options"] = dict()

        p["test_data_name"] = p["data_name"]
        p["test_data_options"] = p["data_options"]

        # network names
        if p["data_name"] in ("mnist", ):
            p["latent_dim"] = 10
            p["encoder_name"] = "PLACE_HODLER"
            p["recon_name"] = "bernoullix"
        else:
            p["latent_dim"] = 128
            p["encoder_name"] = "PLACE_HODLER"
            p["recon_name"] = "guassian_in_01"

        p["decoder_name"] = p["encoder_name"] # by default, use the paired encoder and decoder

        # latent name
        p["decoder_options"] = dict()
        p["recon_options"] = dict()

        if p["recon_name"] in {"gaussian_in_01", "gaussian_fixedvar_in_01"}:
            p.set(
                "recon_options",
                {
                    **{"stddev_lower_bound": 0.05},  # 0.005
                    **p["recon_options"]
                }
            )

        p["recon_weight"] = 1.0
        p["encoder_options"] = dict()

        p["condition_list"] = []

    @staticmethod
    def model(p):
        p.require("main_model")


# default options for the pipeline
class PipelineOptionDefinition(ModelOptionDefinition, TrainDefOptions):

    @staticmethod
    def train(p):

        p.require("model")
        p.include("trainer")

        # big datasets
        if p["data_name"] in (
            "CelebA", "celeba_80x80_landmark",
            "human_128x128", "human_128x128_landmark"
        ):
            assert p["train_shuffle_capacity"] != "full", "should not use full train_shuffle_capacity for large dataset"


        p["train_subset"] = "train"

        p["data_class_list"] = None

        # preprocessing
        if p["data_name"] in ("mnist",):
            p["train_color_jittering"] = False

    @staticmethod
    def test(p):

        p["rotate_batch_samples"] = False

        p["test_subset"] = "test"
        p["test_limit"] = None
        p.require("train")
        p["test_batch_size"] = p["train_batch_size"]

    @staticmethod
    def all(p):

        p.require("model")
        p.require("train")
        p.require("test")

        p.finalize()

