#!/bin/bash

ls -d var/results/*/test.final/prior_samples var/results/*/test.snapshot/*/prior_samples | \
    while read line; do
        if [ -e $line".png" ]; then continue; fi
        echo "$line"
        montage -mode concatenate `ls $line/*.png` $line".png"
    done

