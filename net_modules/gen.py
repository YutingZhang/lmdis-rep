import importlib
from zutils.py_utils import call_func_with_ignored_args


def get_class_instance(net_type, net_name, cls_name, *args, **kwargs):
    """ Get an instance of the network class

    :param net_type: 'encoder', 'latent', 'decoder', 'recon'
    :param net_name: a string indicating which net func to use
    :param cls_name: a string indicating which class to use
    :return:
    """
    mod_name = "nets."+net_type+"."+net_name
    mod_spec = importlib.util.find_spec(mod_name)
    if mod_spec is None:
        return None
    cls_mod = importlib.import_module(mod_name)
    if hasattr(cls_mod, cls_name):
        cls_type = getattr(cls_mod, cls_name)
        cls_instance = call_func_with_ignored_args(cls_type, *args, **kwargs)
        return cls_instance
    else:
        return None


def try_get_net_factory(net_type, net_name, *args, **kwargs):
    """ Get an instance of the network factory class

    :param net_type: 'encoder', 'latent', 'decoder', 'recon'
    :param net_name: a string indicating which net func to use
    :return:
    """
    return get_class_instance(net_type, net_name, "Factory", *args, **kwargs)
 

def get_net_factory(net_type, net_name, *args, **kwargs):
    """ Get an instance of the network factory class

    :param net_type: 'encoder', 'latent', 'decoder', 'recon'
    :param net_name: a string indicating which net func to use
    :return:
    """
    a = try_get_net_factory(net_type, net_name, *args, **kwargs)
    assert a is not None, "Cannot find such a factory"
    return a


def try_get_net_instance(net_type, net_name, *args, **kwargs):
    """ Get an instance of the network class

    :param net_type: 'data'
    :param net_name: a string indicating which net func to use
    :return:
    """
    return get_class_instance(net_type, net_name, "Net", *args, **kwargs)
 

def get_net_instance(net_type, net_name, *args, **kwargs):
    """ Get an instance of the network class

    :param net_type: 'data'
    :param net_name: a string indicating which net func to use
    :return:
    """
    a = try_get_net_instance(net_type, net_name, *args, **kwargs)
    assert a is not None, "Cannot find such a net"
    return a

