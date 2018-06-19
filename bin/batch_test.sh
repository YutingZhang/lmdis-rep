#!/bin/bash

if [ "$1" == "" ]; then
    echo 'Usage: '$0' [OVERRIDE_PARAM [TEST_PATH [SNAPSHOT_ITER]]]'
    exit -1
fi

SCRIPT_DIR=`dirname "$0"`
TOOLBOX_DIR=`readlink -f "$SCRIPT_DIR"/..`

OVERRIDE_PARAM="$1"
TEST_PATH="$2"
SNAPSHOT_ITER="$3"

if type "sponge" > /dev/null; then
    EDITOR=sponge
else
    EDITOR=cat
fi

$EDITOR | python3 "$TOOLBOX_DIR/tools/run_test_in_folder.py" - "$OVERRIDE_PARAM" "$TEST_PATH" "$SNAPSHOT_ITER"

