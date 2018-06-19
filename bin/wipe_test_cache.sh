#!/usr/bin/env bash

SCRIPT_PATH=`readlink -f "$0"`
SCRIPT_FOLDER=`dirname "$SCRIPT_PATH"`
TOOLBOX_FOLDER=`dirname "$SCRIPT_FOLDER"`
CURRENT_FOLDER=`readlink -f .`

RESULT_FOLDER=$1
if [ -z "$RESULT_FOLDER" ]; then
    if [[ "${CURRENT_FOLDER}/" == "$TOOLBOX_FOLDER"/* ]]; then
        RESULT_FOLDER="$TOOLBOX_FOLDER"/var/results
    else
        RESULT_FOLDER="${CURRENT_FOLDER}"
    fi
fi

TMP_FOLDER=/tmp/clean_zyt_tf_cache.$$
mkdir -p "$TMP_FOLDER"
mkfifo "$TMP_FOLDER"/model_iters
mkfifo "$TMP_FOLDER"/test_iters
mkfifo "$TMP_FOLDER"/rm_iters

echo cd "$RESULT_FOLDER"
cd "$RESULT_FOLDER"

find . -type d -name "test.snapshot" | while read SNAPSHOT_ROOT; do
    echo "$SNAPSHOT_ROOT"
    EXP_ROOT=`dirname "$SNAPSHOT_ROOT"`
    MODEL_ROOT="$EXP_ROOT"/model
    ( cd "$MODEL_ROOT" && ls -d snapshot_step_*.index ) 2>/dev/null | sed -e 's/^snapshot_step_\(.*\)\.index$/\1/' | sort -u > "$TMP_FOLDER"/model_iters &
    ( cd "$SNAPSHOT_ROOT" && ls -d step_* ) 2>/dev/null | sed -e 's/^step_\(.*\)$/\1/' | sort -nu | head -n -1 | sort > "$TMP_FOLDER"/test_iters &
    comm -13 "$TMP_FOLDER"/model_iters "$TMP_FOLDER"/test_iters > "$TMP_FOLDER"/rm_iters &
    cat "$TMP_FOLDER"/rm_iters | while read RM_ITER; do
        echo "$SNAPSHOT_ROOT"/step_"$RM_ITER"
    done | xargs rm -rf
done

rm -r "$TMP_FOLDER"
