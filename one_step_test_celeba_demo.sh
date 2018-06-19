#!/bin/bash

SNAPSHOT_ITER=""
SPECIFIC_MODEL_DIR=$(cd `dirname $0`; pwd)'/pretrained_results/celeba_10'

python3 "./tools/run_test_in_folder.py" "$SPECIFIC_MODEL_DIR" "'test_subset':'demo', 'test_limit':None" "test.demo" "$SNAPSHOT_ITER" "False" "True"
 
TEST_PRED_FILE=$SPECIFIC_MODEL_DIR'/test.demo/posterior_param.mat'
matlab -nosplash -r "tmp=load('$TEST_PRED_FILE'); cd demo; vppAutoKeypointShow(tmp.data, tmp.encoded.structure_param,'output');exit()"
