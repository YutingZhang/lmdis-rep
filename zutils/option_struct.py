from collections import namedtuple
from copy import copy
from easydict import EasyDict as edict


class OptionStruct_UnsetCacheNone:
    pass


class OptionStruct:

    def __init__(self, option_dict_or_struct):
        self.user_dict = {}
        self.enabled_dict = {}
        self.unset_set = set()
        self.option_def = None
        self.option_name = None
        if option_dict_or_struct is not None:
            self.add_user_dict(option_dict_or_struct)

    def add_user_dict(self, option_dict_or_struct):
        if isinstance(option_dict_or_struct, dict):
            app_user_dict = option_dict_or_struct
        elif isinstance(option_dict_or_struct, OptionStruct):
            app_user_dict = option_dict_or_struct.enabled_dict
        else:
            raise ValueError("Invalid option dict")
        self.user_dict = {**self.user_dict, **app_user_dict}

    def set(self, key, value):
        self.enabled_dict[key] = value
        self.user_dict[key] = value

    def unset(self, key):
        self.user_dict.pop(key, None)
        self.enabled_dict.pop(key, OptionStruct_UnsetCacheNone())
        self.unset_set.add(key)

    def set_default(self, key, default_value):
        if key in self.user_dict:
            self.enabled_dict[key] = self.user_dict[key]
        else:
            self.enabled_dict[key] = default_value

    def get_enabled(self, key):
        return self.enabled_dict[key]

    def __getitem__(self, item):
        return self.get_enabled(item)

    def __setitem__(self, key, value):
        self.set_default(key, value)

    def get_namedtuple(self, tuple_type_name=None):
        if tuple_type_name is None:
            assert self.option_name is not None, "tuple_type_name must be specified"
            tuple_type_name = self.option_name
        return namedtuple(tuple_type_name, self.enabled_dict.keys())(**self.enabled_dict)

    def get_dict(self):
        return self.enabled_dict

    def get_edict(self):
        return edict(self.enabled_dict)

    def _require(self, option_name, is_include):
        assert isinstance(self.option_def, OptionDef), "invalid option_def"
        p = self.option_def[option_name]
        self.enabled_dict = {**self.enabled_dict, **p.enabled_dict}
        if is_include:
            self.user_dict = {**self.user_dict, **p.user_dict}
        else:
            self.user_dict = {**self.user_dict, **p.enabled_dict}
        self.unset_set = self.unset_set.union(p.unset_set)

    def include(self, option_name):
        self._require(option_name, is_include=True)

    def require(self, option_name):
        self._require(option_name, is_include=False)

    def finalize(self, error_uneaten=True):
        uneaten_keys = set(self.user_dict.keys()) - set(self.enabled_dict.keys()) - self.unset_set
        if len(uneaten_keys) > 0:
            print("WARNING: uneaten options")
            for k in uneaten_keys:
                print("  %s: " % k, end="")
                print(self.user_dict[k])
            if error_uneaten:
                raise ValueError("uneaten options: " + k)


class OptionDef:

    def __init__(self, user_dict={}, def_cls_or_obj=None):
        self._user_dict = user_dict
        self._opts = {}
        if def_cls_or_obj is None:
            self._def_obj = self
        elif def_cls_or_obj is type:
            self._def_obj = def_cls_or_obj()
        else:
            self._def_obj = def_cls_or_obj

    def __getitem__(self, item):
        if item in self._opts:
            return self._opts[item]
        else:
            assert hasattr(self._def_obj, item), "no such method for option definition"
            p = OptionStruct(self._user_dict)
            p.option_def = self
            p.option_name = item + "_options"
            # opt_def_func = getattr(self._def_obj, item)
            # opt_def_func(p)
            eval("self._def_obj.%s(p)" % item)
            self._opts[item] = p
            return p
