import prettytensor as pt


def default_phase():
    defaults = pt.pretty_tensor_class._defaults
    if 'phase' in defaults:
        return defaults['phase']
    else:
        return pt.Phase.test
