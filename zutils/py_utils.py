from collections import namedtuple
from collections import OrderedDict
import time
import inspect
from copy import copy
from collections import deque
import os
import datetime
from inspect import isfunction, ismethod


def time_stamp_str():
    return datetime.datetime.now().strftime('%Y-%m/%d-%H:%M:%S.%f') + "]"


def convert2set(a):
    if isinstance(a, set):
        return a
    if a is None:
        return set()
    if isinstance(a, list) or isinstance(a, tuple):
        return set(a)
    else:
        return {a}


def dict2namedtuple(d, tuple_name=None):
    if tuple_name is None:
        tuple_name = "lambda_namedtuple"
    return namedtuple(tuple_name, d.keys())(**d)


def recursive_generic_condition_func(x, *args):
    return not isinstance(x, (list, tuple, dict))


class RecursiveApplyRemove:
    pass


def recursive_apply_removing_tag():
    return RecursiveApplyRemove


def _recursive_apply(condition_func, func, *args, **kwargs):

    def generic_backup_func(*my_args):
        if len(my_args) == 1:
            return my_args[0]
        else:
            return my_args

    if condition_func is None:
        condition_func = recursive_generic_condition_func

    backup_func = kwargs.pop("backup_func", generic_backup_func)
    assert not kwargs, "unrecognized args"

    if condition_func(*args):
        return func(*args)
    elif isinstance(args[0], (list, tuple)):
        L = list()
        for i in range(len(args[0])):
            sub_args = (t[i] for t in args)
            elt_val = _recursive_apply(condition_func, func, *sub_args, backup_func=backup_func)
            if elt_val is not RecursiveApplyRemove:
                L.append(elt_val)
        if not L and args[0]:
            return RecursiveApplyRemove
        atype = type(args[0])
        if atype is list or atype is tuple:
            return atype(L)
        else:
            try:
                return atype(*L)
            except ValueError:
                pass
            return atype(L)
    elif isinstance(args[0], dict):
        D = type(args[0])()
        for k in args[0]:
            sub_args = []
            for t in args:
                if k in t:
                    sub_args.append(t[k])
                else:
                    sub_args.clear()
                    break
            if sub_args:
                elt_val = _recursive_apply(condition_func, func, *sub_args, backup_func=backup_func)
            else:
                # if key does not exist in every dict, then simply keep the first
                elt_val = args[0][k]
            if elt_val is not RecursiveApplyRemove:
                D[k] = elt_val
        if not D and args[0]:
            return RecursiveApplyRemove
        return D
    else:
        return backup_func(*args)


def recursive_apply(condition_func, func, *args, **kwargs):
    a = _recursive_apply(condition_func, func, *args, **kwargs)
    if a is RecursiveApplyRemove:
        a = type(args[0])()
    return a


def recursive_flatten_to_list(condition_func, x):

    if condition_func is None:
        condition_func = recursive_generic_condition_func

    if condition_func(x):
        return [x]
    elif isinstance(x, (list, tuple)):
        return [z for y in x for z in recursive_flatten_to_list(condition_func, y)]
    elif isinstance(x, dict):
        return [z for y in x.values() for z in recursive_flatten_to_list(condition_func, y)]
    else:
        return []


def recursive_flatten_with_wrap_func(condition_func, x):

    if condition_func is None:
        condition_func = recursive_generic_condition_func

    return (recursive_flatten_to_list(condition_func, x),
            lambda val: recursive_wrap(condition_func, val, x))


class _RecursiveWrapIDT:    # index tracker
    def __init__(self):
        self.i = 0

    def inc(self):
        self.i += 1


def recursive_wrap(condition_func, val, ref):
    if condition_func is None:
        condition_func = recursive_generic_condition_func
    return _recursive_wrap(condition_func, val, ref, _RecursiveWrapIDT())


def _recursive_wrap(condition_func, val, ref, idt):

    if condition_func(ref):
        cur_id = idt.i
        idt.inc()
        return val[cur_id]
    elif isinstance(ref, (list,tuple)):
        L = list()
        for t in ref:
            L.append(_recursive_wrap(condition_func, val, t, idt))
        atype = type(ref)
        if atype is list or atype is tuple:
            return atype(L)
        else:
            try:
                return atype(*L)
            except ValueError:
                pass
            return atype(L)
    elif isinstance(ref, dict):
        D = type(ref)()
        for k, v in ref.items():
            D[k] = _recursive_wrap(condition_func, val, v, idt)
        return D
    else:
        return ref


def first_element_apply(condition_func, func, *args):
    if condition_func is None:
        condition_func = recursive_generic_condition_func
    if condition_func(*args):
        return func(*args)
    elif isinstance(args[0], (list,tuple)):
        for i in range(len(args[0])):
            sub_args = (t[i] for t in args)
            v = first_element_apply(condition_func, func, *sub_args)
            if v is not None:
                return v
        return None
    elif isinstance(args[0], dict):
        for k in args[0]:
            sub_args = (t[k] for t in args)
            v = first_element_apply(condition_func, func, *sub_args)
            if v is not None:
                return v
        return None
    else:
        return None


def flatten_str_dict(hierarchical_dict):
    flatten_dict = OrderedDict()
    for k, v in hierarchical_dict.items():
        if isinstance(v, dict):
            flatten_v = flatten_str_dict(v)
            for kk, vv in flatten_v.items():
                flatten_dict[k + kk] = vv
        else:
            flatten_dict[k] = v
    return flatten_dict


def recursive_indicators(condition_func, x, default_indicator=False):
    if condition_func is None:
        condition_func = recursive_generic_condition_func
    the_indicators = recursive_apply(
        condition_func, lambda _: default_indicator, x, backup_func=lambda _: default_indicator)
    return the_indicators


def recursive_select(x, the_indicators):
    def selector_func(ind, elt_val):
        if ind:
            return elt_val
        else:
            return recursive_apply_removing_tag()
    selected_struct = _recursive_apply(None, selector_func, the_indicators, x)
    if selected_struct is RecursiveApplyRemove:
        if isinstance(x, (list, tuple, dict)):
            selected_struct = type(x)()
        else:
            selected_struct = None
    return selected_struct


def recursive_merge_2dicts(d1, d2):

    k1 = set(d1.keys())
    k2 = set(d2.keys())
    k_both = k1.intersection(k2)
    k1_only = k1-k2
    k2_only = k2-k1

    q = dict()
    for k in k1_only:
        q[k] = d1[k]

    for k in k2_only:
        q[k] = d2[k]

    for k in k_both:
        if isinstance(d1[k], dict) and isinstance(d2[k], dict):
            q[k] = recursive_merge_2dicts(d1[k], d2[k])
        else:
            q[k] = d2[k]

    return q


def recursive_merge_dicts(*args):
    if not args:
        return dict()

    q = args[0]
    for p in args[1:]:
        q = recursive_merge_2dicts(q, p)
    return q


class IfTimeout:

    def __init__(self, timeout):
        self.start_time = time.time()
        self.ignored_time = 0.
        if timeout is None:
            self.target_time = None
        else:
            self.target_time = self.start_time + timeout
        self.interval = None

    def is_timeout(self):
        if self.target_time is None:
            return False
        else:
            cur_time = time.time()
            if cur_time - self.target_time - self.ignored_time > 0:
                if self.interval is None:
                    self.interval = cur_time - self.start_time - self.ignored_time
                return True
            else:
                return False

    def add_ignored_time(self, time_amount):
        self.ignored_time += time_amount


class PeriodicRun:

    def __init__(self, interval, func):
        self.interval = interval
        self.func = func
        self.countdown = None
        self.extra_true_conditions = []
        self.reset()

    def add_extra_true_condition(self, extra_true_condition, extra_func=None):
        if extra_true_condition is None:
            def extra_true_condition_default(**kwargs): return False
            extra_true_condition = extra_true_condition_default
        if extra_func is None:
            extra_func = self.func
        self.extra_true_conditions.append((extra_true_condition, extra_func))

    def add_ignored_time(self, time_amount):
        self.countdown.add_ignored_time(time_amount)

    def reset(self):
        self.countdown = IfTimeout(timeout=self.interval)

    def _run(self, *args, **kwargs):
        output = args[0](*args[1:], **kwargs)
        self.reset()
        return True, output

    def _run_if_timeout(self, *args, **kwargs):
        if self.countdown.is_timeout():
            return self._run(*args, **kwargs)
        else:
            return False, None

    def run_if_timeout(self, *args, **kwargs):
        return self.run_if_timeout_with_prefixfunc(lambda **kwargs: None, *args, **kwargs)

    def run_if_timeout_with_prefixfunc(self, *args, **kwargs):
        for cond, func in reversed(self.extra_true_conditions):   # later added, have higher priority
            if cond():
                args[0]()
                return self._run(func, *args[1:], **kwargs)

        def the_func(*args, **kwargs):
            args[0]()
            return self.func(*args[1:], **kwargs)
        return self._run_if_timeout(the_func, *args, **kwargs)


class FlagEater:

    def __init__(self, flags=None, default_value=False, default_pos_value=True):
        if isinstance(flags, FlagEater):
            self._flags = flags._flags
            self._default_value = flags._default_value
            self._default_pos_value = flags._default_pos_value
        else:
            self._default_value = default_value
            self._default_pos_value = default_pos_value
            self._flags = dict()
            self.add_flags(flags)

    def add_flags(self, flags):
        if flags is None:
            return
        if isinstance(flags, dict):
            self._flags = {**self._flags, **flags}
        elif isinstance(flags, (list, tuple, set)):
            for k in flags:
                self._flags[k] = self._default_pos_value
        else:
            self._flags[flags] = self._default_pos_value

    def pop(self, key):
        return self._flags.pop(key, self._default_value)

    def finalize(self):
        assert not self._flags, "Not all flags are eaten"


def call_func_with_ignored_args(func, *args, **kwargs):

    remaining_args = deque(args)
    remaining_kwargs = copy(kwargs)
    actual_args = list()
    actual_kwargs = OrderedDict()
    for k, v in inspect.signature(func).parameters.items():
        value_has_been_set = False
        if v.kind == v.POSITIONAL_ONLY:
            if remaining_args:
                actual_args.append(remaining_args.popleft())
                value_has_been_set = True
        elif v.kind == v.POSITIONAL_OR_KEYWORD:
            if remaining_args:
                actual_args.append(remaining_args.popleft())
                value_has_been_set = True
            elif remaining_kwargs and k in remaining_kwargs:
                actual_kwargs[k] = remaining_kwargs.pop(k)
                value_has_been_set = True
        elif v.kind == v.VAR_POSITIONAL:
            actual_args.extend(remaining_args)
            value_has_been_set = True
        elif v.kind == v.KEYWORD_ONLY:
            if remaining_kwargs and k in remaining_kwargs:
                actual_kwargs[k] = remaining_kwargs.pop(k)
                value_has_been_set = True
        elif v.kind == v.VAR_KEYWORD:
            actual_kwargs = {**actual_kwargs, **remaining_kwargs}
            value_has_been_set = True
        else:
            raise ValueError("Internal errors: unrecognized parameter kind")

        if not value_has_been_set:
            assert v.default != v.empty, "necessary argument is missing: %s" % v.name

    return func(*actual_args, **actual_kwargs)


def first_in_dict(d):
    for k in d:
        return d[k]


def robust_index(a, i):
    if a is None:
        return None
    else:
        return a[i]


def rbool(a):
    try:
        c = bool(a)
    except OSError:
        raise
    except:
        c = True
    return c


def path_full_split(p):
    s = list()
    a = p
    if not a:
        return s
    if a[-1] == "/":
        s.append("")
        while a and a[-1] == "/":
            a = a[:-1]
    while True:
        [a, b] = os.path.split(a)
        if b:
            s.append(b)
        else:
            if a:
                s.append(a)
            else:
                break
    s.reverse()
    return s


def even_partition(total_num, partition_num):
    smaller_size = total_num//partition_num
    larger_num = total_num - smaller_size * partition_num
    smaller_num = partition_num - larger_num
    p = [smaller_size+1] * larger_num + [smaller_size] * smaller_num
    return p


def even_partition_indexes(total_num, partition_num):
    subset_num = even_partition(total_num, partition_num)
    epi = list()
    for i in range(len(subset_num)):
        epi.extend([i] * subset_num[i])
    return epi


def value_class_for_with(init_value=None):

    class value_for_with:

        current_value = init_value

        def __init__(self, value):
            self._value = value
            self._upper_value = None

        def __enter__(self):
            self._upper_value = value_for_with.current_value
            value_for_with.current_value = self._value
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            value_for_with.current_value = self._upper_value
            return False

    return value_for_with


class dummy_class_for_with:

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def get_new_members(inherent_class, base_class):
    assert issubclass(inherent_class, base_class), "must be inherent class and base class"
    target_mem = dict(inspect.getmembers(inherent_class))
    base_mem = dict(inspect.getmembers(base_class))
    new_mem = list(filter(lambda a: a[0] not in base_mem or base_mem[a[0]] is not a[1], target_mem.items()))
    return new_mem


def update_class_def_per_ref(target_class, ref_class, target_base_class=None, ref_base_class=None):
    # this is for merge two class definition branch into one
    if target_base_class is None:
        target_base_class = object

    reserved_mem_names = set(dict(get_new_members(target_class, target_base_class)).keys())

    if ref_base_class is None:
        ref_base_class = object

    new_mem = dict(get_new_members(ref_class, ref_base_class))
    override_mem_names = set(new_mem.keys()) - set(reserved_mem_names)

    for k in override_mem_names:
        setattr(target_class, k, new_mem[k])


def link_with_instance(self, another):
    # this is for merges the defintion of another instance

    my_attr_dict = list(filter(
        lambda kk: not (kk.startswith('__') and kk.endswith('__')),
        dir(self)))

    for k in dir(another):
        if k.startswith('__') and k.endswith('__'):
            continue
        if k in my_attr_dict:
            continue
        v = getattr(another, k)
        if not (isfunction(v) or ismethod(v) or callable(v)):
            continue
        setattr(
            self, k, (lambda vv: lambda *arg, **kwargs: vv(*arg, **kwargs))(v))


class ArgsSepFunc:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.my_args = args
        self.my_kwargs = kwargs

    def set_args(self, *args, **kwargs):
        self.my_args = args
        self.my_kwargs = kwargs

    def __call__(self):
        return self.func(*self.my_args, **self.my_kwargs)

