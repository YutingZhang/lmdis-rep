import sys
import os
import pickle
from zutils.py_utils import *
from model.pipeline import Pipeline


def run_test(folder_path, override_dict, test_path, snapshot_iter, is_large, save_img_data):

    print("Folder path: %s" % folder_path)

    with open(os.path.join(folder_path, "PARAM.p"), 'rb') as f:
        opt0 = pickle.load(f)

    # opt = {**opt0, **override_dict}
    opt = recursive_merge_dicts(opt0, override_dict)

    vp = Pipeline(
        None, opt, model_dir=folder_path,
        auto_save_hyperparameters=False, use_logging=False
    )

    print(vp.opt)
    with vp.graph.as_default():
        sess = vp.create_session()
        vp.run_full_test_from_checkpoint(sess, test_path=test_path, snapshot_iter=snapshot_iter, is_large=is_large, save_img_data=save_img_data)


def main():
    if not sys.argv:
        print("Usage: run_test_in_folder.py EXP_PATH [OVERRIDE_PARAM [TEST_PATH [SNAPSHOT_ITER]]]")
        exit(-1)
    folder_path = sys.argv[1]

    opt_command = sys.argv[2] if len(sys.argv) > 2 else ""
    override_dict = eval("{%s}" % opt_command)

    test_path = sys.argv[3] if len(sys.argv) > 3 else ""
    if not test_path:
        test_path = None

    snapshot_iter = sys.argv[4] if len(sys.argv) > 4 else ""
    if not snapshot_iter:
        snapshot_iter = None
    else:
        snapshot_iter = int(snapshot_iter)

    is_large = sys.argv[5] if len(sys.argv)>5 else ""
    if is_large == 'False':
        is_large = False
    elif is_large == 'True':
        is_large = True
    else:
        is_large = False

    save_img_data = sys.argv[6] if len(sys.argv)>6 else ""
    if save_img_data == 'False':
        save_img_data = False
    elif save_img_data == 'True':
        save_img_data = True
    else:
        save_img_data = True

    if folder_path != "-":
        run_test(folder_path, override_dict, test_path, snapshot_iter, is_large, save_img_data)
    else:
        print("Please Input a List of Experiment Folders:")
        for line in sys.stdin:
            line = line[:-1]
            if not line:
                continue
            run_test(line, override_dict, test_path, snapshot_iter, is_large, save_img_data)


if __name__ == "__main__":
    main()
