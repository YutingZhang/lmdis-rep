import tensorflow as tf
from tensorflow.python.lib.io import file_io
import stat
from collections import OrderedDict, Iterable
from copy import copy
from types import MethodType
import numpy as np
import re
import os

from zutils.py_utils import path_full_split, value_class_for_with

from tensorflow.python.client import device_lib

# ----------------------------------------------------------------------------------------------------------------


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    assert len(gpus)<=1, \
        "ERROR: this code does not support multiple devices. Use CUDA_VISIBLE_DEVICES=... to specify the GPU."
    return gpus


def current_scope_name():
    dummy_scope_name = "_dummy"
    with tf.variable_scope(dummy_scope_name) as scope:
        the_scope_name = scope.name[:-len(dummy_scope_name)]

    if the_scope_name[-1] == "/":
        the_scope_name = the_scope_name[:-1]
    return the_scope_name


def get_existing_item(var_full_name, collection_name):
    var_list = tf.get_collection(collection_name)
    for v in var_list:
        if v.name == var_full_name:
            return v
    return None


def get_existing_variable(var_full_name):
    return get_existing_item(var_full_name, tf.GraphKeys.GLOBAL_VARIABLES)


# ----------------------------------------------------------------------------------------------------------------


def graph_keys_dict():
    """ figure out current valid collection names

    :return: 
    """

    graph_keys_dict_raw = dict(tf.GraphKeys.__dict__)
    graph_keys_dict_raw["AUX_LOSS"] = "aux_loss"  # self defined collections
    graph_keys_dict_clean = dict()
    for k, v in graph_keys_dict_raw.items():
        if not (k.startswith('__') and k.endswith('__')):
            if isinstance(v, str):
                graph_keys_dict_clean[k] = v

    return graph_keys_dict_clean


GraphKeysDict = graph_keys_dict()


class GraphCollectionSnapshot:

    def __init__(self, graph=None):

        # basic
        if graph is None:
            graph = tf.get_default_graph()
        self.graph = graph

        # snapshot collections
        self._collections = dict(
            (v, self.graph.get_collection(v)) for v in GraphKeysDict.values())
        self._collections["_operations"] = self.graph.get_operations()

    def get_collection(self, key, scope=None):

        full_collection = self._collections[key]
        if scope is None:
            result_collection = full_collection
        else:
            scope_collection = self.graph.get_collection(key, scope)
            result_collection = list(set(full_collection).intersection(scope_collection))
        return result_collection

    def get_operations(self):
        return self._collections["_operations"]

    def call_set_func(self, func_name, *args, **kwargs):

        def extract_sub_args(key, arg):
            if isinstance(arg, GraphCollectionSnapshot):
                return arg._collections[key]
            else:
                return arg

        output_type = type(self)
        new_collections = dict()
        for col_key, col_value in self._collections.items():
            sub_args = list(extract_sub_args(col_key, vv) for vv in args)
            sub_kwargs = dict((kk, extract_sub_args(col_key, vv)) for kk, vv in kwargs.items())
            sv = set(col_value)
            set_op = getattr(sv, func_name)
            sub_result = set_op(*sub_args, **sub_kwargs)
            if output_type is type(self):
                if isinstance(sub_result, set):
                    sub_result = list(sub_result)
                elif sub_result is None:
                    output_type = None
                else:
                    output_type = type(output_type)
            new_collections[col_key] = sub_result

        if output_type is type(self):
            new_obj = copy(self)
            new_obj._collections = new_collections
            return new_obj
        elif output_type is None:
            return None
        else:
            return new_collections


def graph_collection_snapshot_add_set_funcs():
    # define set funcs
    set_func_list = [
        "copy",
        "difference", "difference_update",
        "intersection", "intersection_update",
        "symmetric_difference", "symmetric_difference_update",
        "union",
        "issubset", "issuperset", "isdisjoint",
        "__and__",
        "__ge__",
        "__isub__",
        "__or__",
        "__gt__",
        "__ixor__",
        "__str__",
        "__le__",
        "__sub__",
        "__iand__",
        "__len__",
        "__lt__",
        "__ror__",
        "__xor__",
        "__eq__",
        "__ne__",
        "__rsub__",
        "__ior__",
        "__rxor__",
    ]
    for sf in set_func_list:
        foo = (lambda f: lambda self, *args, **kwargs: self.call_set_func(f, *args, **kwargs))(sf)
        setattr(GraphCollectionSnapshot, sf, foo)


graph_collection_snapshot_add_set_funcs()


class SubgraphCollectionSnapshots(OrderedDict):

    def __init__(self, graph=None):

        if graph is None:
            graph = tf.get_default_graph()
        self.graph = graph
        self._previous_snapshot = None

    def sub_snapshot(self, key):
        graph_snapshot = GraphCollectionSnapshot(self.graph)
        if self._previous_snapshot is None:
            cur_snapshot = graph_snapshot
        else:
            cur_snapshot = graph_snapshot.difference(self._previous_snapshot)
        self[key] = cur_snapshot
        self._previous_snapshot = graph_snapshot


# ----------------------------------------------------------------------------------------------------------------


def regularization_sum(graph=None, trainable_variables=None, default_scale=1., scope=None, display=False):
    # run it after constructed all training net

    if default_scale is None:
        default_scale = 0.

    if trainable_variables is not None:
        assert graph is None and scope is None, "variables have beeen specified directly"
    else:
        trainable_variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    excluding_names = [
        "/batch_normalize/moving_",
        "/batch_normalize/beta",
        "/bias"
    ]

    # if display:
    #     print(" * Model weight decay: ")

    total_sum = 0.
    var_l2 = OrderedDict()
    for v in trainable_variables:
        var_included = True
        for en in excluding_names:
            if v.name.find(en) >= 0:
                var_included = False
                break

        if var_included:
            the_scale = default_scale
            if hasattr(v, "custom_decay_scale"):
                the_scale = v.custom_decay_scale
            var_included = the_scale>0

        if display:
            if var_included:
                print("   - %s : l2 * %g" % (v.name, the_scale))
            else:
                print("   - %s : none" % v.name)

        if not var_included:
            continue

        v_sum = tf.nn.l2_loss(v)


        total_sum += the_scale * v_sum
        var_l2[v.name] = v_sum

    return total_sum, var_l2

# ----------------------------------------------------------------------------------------------------------------


class SaverImproved(tf.train.Saver):

    # some hacks to Saver
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    assert hasattr(tf.train.Saver, '_delete_file_if_exists'), \
        "not compatible with tf.train.Saver definition. Version conflicts?"

    def _delete_file_if_exists(self, filespec):
        for pathname in file_io.get_matching_files(filespec):
            s = os.stat(pathname)[stat.ST_MODE]
            if not (s & stat.S_IWUSR):      # skip read-only file
                continue
            file_io.delete_file(pathname)


class MultiDeviceSaver(SaverImproved):

    def __init__(self, var_list=None, *args, **kwargs):
        class AuxiliaryInfo:
            def __init__(self):
                self.graph = None
                self.var_list = None
                self.update_shared = None

        self.aux = AuxiliaryInfo()
        self.aux.graph = tf.get_default_graph()

        if var_list is None:
            var_list = self.aux.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # remove shared variables with different names
        var_list = ruleout_shared_vars(var_list)
        # ---------------------------------

        self.aux.var_list = var_list
        self.aux.update_shared = update_shared_vars(self.aux.var_list)

        super().__init__(var_list=var_list, *args, **kwargs)

    @property
    def the_graph(self):
        return self.aux.graph

    @property
    def the_var_list(self):
        return self.aux.var_list

    def restore(self, sess, *args, **kwargs):
        # do ordinary restore to the ps device
        super().restore(sess, *args, **kwargs)
        # sync to shared vars
        sess.run(self.aux.update_shared)
        # sync to workers
        self.sync_to_worker_devices(sess)

    def sync_to_worker_devices(self, sess):
        # copy to worker devices
        restored_vars = self._gen_indexed_var_dict(self.the_var_list)
        all_vars = self._gen_indexed_var_dict(self.the_graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        all_assign = []
        intersected_vars = set(all_vars.keys()).intersection(set(restored_vars.keys()))
        for k in intersected_vars:
            all_l = all_vars[k]
            restored_l = restored_vars[k]
            if len(all_l) <= len(restored_l):
                continue
            for i in range(len(restored_l), len(all_l)):
                all_assign.append(tf.assign(all_l[i], restored_l[i % len(restored_l)]))
        sess.run(all_assign)

    @staticmethod
    def _gen_indexed_var_dict(var_list):

        indexed_var_dict = dict()

        for v in var_list:
            s = v.name
            if s.endswith(":0"):
                s = s[:-2]
            else:
                raise ValueError("Unrecognized variable name: %s" % s)

            var_index_matched = re.findall('_[0-9]+$', s)

            if var_index_matched:
                var_index = int(var_index_matched[0][1:])
                s = s[:-len(var_index_matched)]
            else:
                var_index = 0

            if s not in indexed_var_dict:
                indexed_var_dict[s] = []

            l = indexed_var_dict[s]
            old_l_len = len(l)
            if old_l_len == var_index:
                l.append(v)
            elif old_l_len > var_index:
                l[var_index] = v
            else:
                l += [None]*(var_index-old_l_len)
                l.append(v)
        return indexed_var_dict

    @staticmethod
    def _gen_var_dict(var_list, ignored_scope_level=0):
        var_dict = dict()
        for v in var_list:
            nm = v.name
            if ignored_scope_level>0:
                nm_split = path_full_split(nm)
                if len(nm_split)>ignored_scope_level:
                    new_nm_split = nm_split[ignored_scope_level:]
                else:
                    new_nm_split = nm_split[-1:]
                nm = os.path.join(*new_nm_split)
            assert nm not in var_dict, \
                "conflict variable names"
            var_dict[nm] = v
        return var_dict

    def load_weights(self, sess, snapshot_file_path, ignored_scope_level=0, display=True, var_list=None):

        def nprint(*args, **kwargs):
            if display:
                print(*args, **kwargs)

        # if is folder
        if os.path.exists(snapshot_file_path) and os.path.isdir(snapshot_file_path):
            ckpt = tf.train.get_checkpoint_state(snapshot_file_path)
            assert ckpt and ckpt.model_checkpoint_path, \
                "no checkpoint file is available"
            snapshot_file_path = ckpt.model_checkpoint_path

        # load meta graph
        assert os.path.exists(snapshot_file_path + ".meta"), \
            "Cannot find the meta graph file"
        nprint(" - load meta-graph")
        snapshot_graph = tf.Graph()
        with snapshot_graph.as_default(), tf.device("/cpu:0"):
            snapshot_saver_all = tf.train.import_meta_graph(
                snapshot_file_path+".meta", clear_devices=True
            )

        snapshot_var_list = snapshot_graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        if var_list is None:
            my_var_list = self.the_var_list
        else:
            my_var_list = var_list
            assert all(v in self.the_var_list for v in var_list), "some specified vars are not "

        nprint(" - variable list:")

        snapshot_var_dict = self._gen_var_dict(snapshot_var_list, ignored_scope_level=ignored_scope_level)
        my_var_dict = self._gen_var_dict(my_var_list, ignored_scope_level=ignored_scope_level)
        valid_var_names = set(my_var_dict.keys()).intersection(set(snapshot_var_dict.keys()))
        no_source_var_list = [my_var_dict[k] for k in set(my_var_dict.keys()).difference(valid_var_names)]
        no_target_var_list = [snapshot_var_dict[k] for k in set(snapshot_var_dict.keys()).difference(valid_var_names)]

        for v in no_source_var_list:
            nprint("  - [no source]: %s" % v.name)
        for v in no_target_var_list:
            nprint("  - %s: [no target]" % v.name)

        source_var_list = list()
        target_var_list = list()
        for k in valid_var_names:
            v_src = snapshot_var_dict[k]
            v_tgt = my_var_dict[k]
            nprint("  - %s -> %s" % (v_src.name, v_tgt.name))
            source_var_list.append(v_src)
            target_var_list.append(v_tgt)

        nprint(" - Load valid variables")
        with snapshot_graph.as_default(), tf.device("/cpu:0"):
            src_sess = tf.Session()
            snapshot_saver = tf.train.Saver(var_list=source_var_list)
            snapshot_saver.restore(src_sess, snapshot_file_path)

        nprint(" - Read valid variables from the source graph")
        source_var_values = src_sess.run(source_var_list)

        nprint(" - Clean up checkpoint session")
        # clear the restored graph
        src_sess.close()
        del src_sess
        # with snapshot_graph.as_default():
        #     tf.reset_default_graph()

        nprint(" - Write valid variables into the main graph")
        assign_list = list()
        with self.the_graph.as_default(), tf.device("/cpu:0"):
            for v_tgt, v_val in zip(target_var_list, source_var_values):
                assign_list.append(tf.assign(v_tgt, v_val))
        sess.run(assign_list)

        nprint(" - Weight loading is done")


# ----------------------------------------------------------------------------------------------------------------

def pair_vars_between_scope(src, dst, src_vars=None, dst_vars=None):
    def canonicalize_scope_name(s):
        if isinstance(s, tf.VariableScope):
            s = s.name
        return s + "/"

    def canonicalize_vars(vars, scope_path):
        if vars is None:
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        vd = dict()
        prefix_len = len(scope_path)
        for v in vars:
            if v.name.startswith(scope_path):
                vd[v.name[prefix_len:]] = v
        return vd

    src = canonicalize_scope_name(src)
    dst = canonicalize_scope_name(dst)

    src_vars = canonicalize_vars(src_vars, src)
    dst_vars = canonicalize_vars(dst_vars, dst)

    assert len(dst_vars) == len(src_vars) and all(k in dst_vars for k in src_vars), \
        "variables mismatches"

    pair_list = []
    for k, src_v in src_vars.items():
        pair_list.append((src_v, dst_vars[k]))    # (src, dst)
    return pair_list


def copy_vars_per_pair_list(pair_list):
    assign_list = []
    for src_v, dst_v in pair_list:
        assign_list.append(tf.assign(dst_v, src_v))
    return tf.group(*assign_list)


def copy_vars_between_scope(src, dst, src_vars=None, dst_vars=None):
    pair_list = pair_vars_between_scope(src=src, dst=dst, src_vars=src_vars, dst_vars=dst_vars)
    return copy_vars_per_pair_list(pair_list)


def ruleout_shared_vars(var_list):
    shared_pair_list = tf.get_collection("variable_sharing")
    var_list = set(var_list) - set(dst_v for _, dst_v in shared_pair_list)
    var_list = list(var_list)
    return var_list


def update_shared_vars(var_list=None):
    shared_pair_list = tf.get_collection("variable_sharing")
    if var_list is not None:
        var_list = set(var_list)
        shared_pair_list = list(filter(lambda x: x[1] in var_list, shared_pair_list))
    return copy_vars_per_pair_list(shared_pair_list)


# ----------------------------------------------------------------------------------------------------------------


def match_scope(item, scope_path):
    if isinstance(item, tf.Variable):
        # exact match
        return item.name.startswith(scope_path)
    else:
        # robust match
        t = path_full_split(item.name)
        r = path_full_split(scope_path)
        if r and not r[-1]:
            r = r[:-1]

        # check if it is possible to match
        if len(t) <= len(r):
            return False

        # test is the first scope can match
        for k in range(len(r)):
            if not t[k].startswith(r[k]):
                return False
            if len(t[k]) > len(r[k]):
                if not t[k][len(r[k])] == "_":
                    return False
                tk_postfix = t[k][len(r[k])+1:]
                try:
                    int(tk_postfix)
                except ValueError:
                    return False
        return True


# ----------------------------------------------------------------------------------------------------------------


EnableAuxLoss = value_class_for_with(True)


def add_to_aux_loss(v, disp_name=None):
    if not EnableAuxLoss.current_value:
        return
    if disp_name is not None:
        v.disp_name = disp_name
    tf.add_to_collection("aux_loss", v)

# ----------------------------------------------------------------------------------------------------------------

def add_to_freeze_collection(vars):
    if not isinstance(vars, (list, tuple)):
        vars = [vars]
    for v in vars:
        tf.add_to_collection("freeze", v)


def get_freeze_collection():
    return tf.get_collection("freeze")


# ----------------------------------------------------------------------------


class ValueSchedulerPool:

    @staticmethod
    def exponential_decay(global_step, *args, **kwargs):
        return tf.train.exponential_decay(args[0], global_step, *args[1:], **kwargs)

    @staticmethod
    def piecewise_constant(global_step, boundaries, values, *args, **kwargs):
        #boundaries = np.array(boundaries).astype(global_step.dtype.as_numpy_dtype)
        #global_step = tf.cast(global_step, tf.as_dtype(np.array(boundaries).dtype))
        return tf.train.piecewise_constant(tf.cast(global_step, "int32"), boundaries, values, *args, **kwargs)

    @staticmethod
    def polynomial_decay(global_step, *args, **kwargs):
        return tf.train.polynomial_decay(args[0], global_step, *args[1:], **kwargs)

    @staticmethod
    def natural_exp_decay(global_step, *args, **kwargs):
        return tf.train.natural_exp_decay(args[0], global_step, *args[1:], **kwargs)

    @staticmethod
    def inverse_time_decay(global_step, *args, **kwargs):
        return tf.train.inverse_time_decay(args[0], global_step, *args[1:], **kwargs)

    @staticmethod
    def linear(global_step, start_value, end_value, end_step, start_step=0):
        if end_step == start_step:
            v = tf.where(global_step<end_value, start_value, end_value)
        else:
            assert end_step > start_step, "end_step should be no less than start_step"
            r = (tf.to_float(global_step)-start_step) / (end_step - start_step)
            r = tf.clip_by_value(r, clip_value_min=0., clip_value_max=1.)
            v = r * end_value + (1-r) * start_value
        return v


class ValueScheduler:

    def __init__(self, scheduler_type, *args, **kwargs):
        assert hasattr(ValueSchedulerPool, scheduler_type), "no such scheduler exists: %s" % scheduler_type
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
            del kwargs["dtype"]
        else:
            dtype = "float32"
        self.type = scheduler_type
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        s = self.type
        for v in self.args:
            s += "_" + str(v).replace(", ", ",")
        for k, v in self.kwargs.items():
            s += "_" + str(k) + "=" + str(v).replace(", ", ",")
        return s

    def __bool__(self):
        return True


def value_scheduler_to_tensor(scheduler_def, global_step=None):
    assert isinstance(scheduler_def, ValueScheduler), "wrong argument"
    if global_step is None:
        global_step = tf.train.get_or_create_global_step()

    assert hasattr(ValueSchedulerPool, scheduler_def.type), \
        "no such scheduler exists: %s" % scheduler_def.type
    scheduler_type = getattr(ValueSchedulerPool, scheduler_def.type)
    if isinstance(scheduler_type, type):
        func = scheduler_type(*scheduler_def.args, **scheduler_def.kwargs)
    else:
        def func(x):
            return scheduler_type(x, *scheduler_def.args, **scheduler_def.kwargs)

    y = func(global_step)
    y = tf.cast(y, scheduler_def.dtype)

    return y


def sequential_data_buffer(source, batch_size, capacity, enqueue_many=False):

    # Create a random shuffle queue.
    if not isinstance(source, (list, tuple)):
        source = (source,)

    queue = tf.FIFOQueue(
        capacity=capacity,
        shapes=tuple((s.shape[1:] if enqueue_many else s.shape) for s in source),
        dtypes=tuple(s.dtype for s in source)
    )

    # Create an op to enqueue one item.
    if enqueue_many:
        enqueue = queue.enqueue_many(source)
    else:
        enqueue = queue.enqueue(source)

    # Create a queue runner that, when started, will launch 4 threads applying
    # that enqueue op.
    num_threads = 1  # must be 1
    qr = tf.train.QueueRunner(queue, [enqueue] * num_threads)

    # Register the queue runner so it can be found and started by
    # `tf.train.start_queue_runners` later (the threads are not launched yet).
    tf.train.add_queue_runner(qr)

    # Create an op to dequeue a batch
    return queue.dequeue_many(batch_size)


def batch_rotating_data_buffer(source, batch_size, capacity, enqueue_many=False):

    # Create a random shuffle queue.
    if not isinstance(source, (list, tuple)):
        source = (source,)

    queue_shape = tuple((s.shape[1:] if enqueue_many else s.shape) for s in source)
    queue_dtype = tuple(s.dtype for s in source)
    inner_queue = tf.FIFOQueue(
        capacity=capacity,
        shapes=queue_shape,
        dtypes=queue_dtype
    )
    outer_queue = tf.FIFOQueue(
        capacity=batch_size*batch_size*2,
        shapes=queue_shape + ([],),
        dtypes=queue_dtype + (tf.bool,)
    )

    # Create an op to enqueue one item.
    if enqueue_many:
        inner_enqueue = inner_queue.enqueue_many(source)
    else:
        inner_enqueue = inner_queue.enqueue(source)

    #inner_enqueue = tf.Print(inner_enqueue, ["inner_enqueue"])

    # dequeue inner
    the_batch_data = inner_queue.dequeue_many(batch_size)

    if not isinstance(the_batch_data, (tuple, list)):
        the_batch_data = (the_batch_data,)
    rotated_batch_data = []
    for data_single in the_batch_data:
        rotated_batch_data.append(tf.tile(
            data_single, [batch_size] + [1] * (len(data_single.shape) - 1)
        ))
    rotated_batch_data.append(tf.constant(
        np.reshape(np.eye(batch_size, dtype=np.int32), [batch_size*batch_size]), dtype=tf.bool))

    outer_enqueue = outer_queue.enqueue_many(rotated_batch_data)

    output_tensor = outer_queue.dequeue_many(batch_size)

    # Create a queue runner that, when started, will launch 4 threads applying
    # that enqueue op.
    num_threads = 1  # must be 1
    inner_qr = tf.train.QueueRunner(inner_queue, [inner_enqueue] * num_threads)
    outer_qr = tf.train.QueueRunner(outer_queue, [outer_enqueue] * num_threads)

    # Register the queue runner so it can be found and started by
    # `tf.train.start_queue_runners` later (the threads are not launched yet).
    tf.train.add_queue_runner(inner_qr)
    tf.train.add_queue_runner(outer_qr)

    # Create an op to dequeue a batch
    return output_tensor


def bare_name(name):
    assert isinstance(name, str), "name must be a str"
    a = name.split("/")
    ff = a[-1]
    b = ff.split(":")
    if len(b) == 1:
        f = ff
    else:
        f = ":".join(b[:-1])
    return f


def name_parts(name):
    assert isinstance(name, str), "name must be a str"
    a = name.split("/")
    ff = a[-1]
    b = ff.split(":")
    if len(b) == 1:
        f = ff
        ext = ""
    else:
        f = ":".join(b[:-1])
        ext = ":" + b[-1]
    p = '/'.join(a[:-1])
    return p, f, ext
