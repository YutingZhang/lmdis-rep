#!/bin/bash

PORT=$1
if [ -z "$PORT" ]; then
    PORT=6006
fi

SCRIPT_DIR=`dirname "$0"`
TOOLBOX_DIR=`readlink -f "$SCRIPT_DIR"/..`
RESULT_DIR="$TOOLBOX_DIR"/var/results

if type "sponge" > /dev/null; then
    EDITOR=sponge
else
    EDITOR=cat
fi

echo "Please Input the List of Experiments:"

$EDITOR | while read line; do
    echo $PORT ":" $line
    if [[ "$str" != /* ]]; then
        line=$RESULT_DIR/$line
    fi
    ( cd "$line"/logs; tensorboard --logdir=. --port=$PORT > /dev/null 2>&1 ) &
    PORT=$((PORT+1))
done

while true; do sleep 1; done

