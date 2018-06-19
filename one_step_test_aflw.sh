#!/bin/bash

SNAPSHOT_ITER=""
SPECIFIC_MODEL_DIR=$(cd `dirname $0`; pwd)'/pretrained_results/aflw_10'

TRAIN_GT_FILE=$(cd `dirname $0`; pwd)'/data/aflw_data/aflw_train_keypoints.mat'
TEST_GT_FILE=$(cd `dirname $0`; pwd)'/data/aflw_data/aflw_test_keypoints.mat'

python3 "./tools/run_test_in_folder.py" "$SPECIFIC_MODEL_DIR" "'test_subset':'test', 'test_limit':None" "test.test" "$SNAPSHOT_ITER" "True" "False"

python3 "./tools/run_test_in_folder.py" "$SPECIFIC_MODEL_DIR" "'test_subset':'train', 'test_limit':None" "test.train" "$SNAPSHOT_ITER" "True" "False"

DATA_DIR=$SPECIFIC_MODEL_DIR'/test.train'
matlab -nodesktop -nosplash -r "cd('./evaluation');merge_large_dataset('$DATA_DIR');exit;"
DATA_DIR=$SPECIFIC_MODEL_DIR'/test.test'
matlab -nodesktop -nosplash -r "cd('./evaluation');merge_large_dataset('$DATA_DIR');exit;"

TRAIN_PRED_FILE=$SPECIFIC_MODEL_DIR'/test.train/posterior_param.mat'
TEST_PRED_FILE=$SPECIFIC_MODEL_DIR'/test.test/posterior_param.mat'
matlab -nosplash -r "cd('./evaluation');face_evaluation('$TRAIN_PRED_FILE','$TRAIN_GT_FILE','$TEST_PRED_FILE','$TEST_GT_FILE');exit()"
