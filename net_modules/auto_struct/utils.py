from zutils.py_utils import recursive_merge_dicts


class ModuleOutputStrip:

    def __init__(self):
        self.extra_outputs = dict()
        self.extra_outputs["save"] = dict()
        self.extra_outputs["extra_recon"] = dict()
        self.extra_outputs["for_decoder"] = dict()
        self.extra_outputs["for_discriminator"] = dict()

    def __call__(self, module_output):
        if isinstance(module_output, tuple):
            if not module_output:
                return None
            if len(module_output) == 1:
                return module_output[0]

            if module_output[-1] is not None:
                self.extra_outputs = recursive_merge_dicts(
                    self.extra_outputs, module_output[-1]
                )
            if len(module_output) == 2:
                return module_output[0]
            else:
                return module_output[:-1]
        else:
            return module_output

