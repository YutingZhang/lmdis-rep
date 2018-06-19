from scipy.io import savemat

import net_modules.gen
from model.pipeline_netdef import PipelineNetDef
from runner.one_epoch_runner import OneEpochRunner
from zutils.py_utils import *

net_factory = net_modules.gen.get_net_factory
net_instance = net_modules.gen.get_net_instance


class Pipeline(PipelineNetDef):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # training ------------------------------------------

    def resume(self, sess):
        self.train.groups["ae"].trainer.run(sess)

    # posterior parameters -----------------------------------------------------------------------

    def posterior_param(self, sess, output_fn, is_large=False, save_img_data=True):
        if save_img_data:
            output_list = self.posterior.outputs
        else:
            output_list = self.posterior.outputs
            output_list.pop('data')
        print(output_list)
        r = OneEpochRunner(
            self.posterior.data_module,
            output_fn = output_fn,
            num_samples=self.opt.test_limit,
            output_list=output_list,
            disp_time_interval=self.opt.disp_time_interval,
            is_large=is_large)
        return r.run(sess)

    """
    def dump_posterior_param(self, pp, output_fn):
        dir_path = os.path.dirname(output_fn)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # pickle.dump(pp, open(output_fn + ".p", "wb"))
        if "vis" in pp["decoded"]:
            pp["decoded"]["vis"] = self.output_scaled_color_image(pp["decoded"]["vis"])
        if "data" in pp:
            pp["data"] = self.output_scaled_color_image(pp["data"])
        savemat(output_fn + ".mat", pp)
    """

    # all test  -----------------------------------------------------------------------------
    def test(self, sess, output_dir, is_snapshot=False, is_large=False, save_img_data=True):

        def nprint(*args, **kwargs):
            if not is_snapshot:
                print(*args, **kwargs)

        # not necessary for snapshot
        nprint('========== Posterior parameters')
        self.posterior_param(sess, os.path.join(output_dir, "posterior_param"), is_large, save_img_data)
        nprint('-- Done')


