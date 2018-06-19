#!/bin/bash

if [ -z "$1" ]; then
    BASE_FOLDER="var/results"
else
    BASE_FOLDER=$1
fi

cd $BASE_FOLDER

rm -f _figure_path.txt
mkfifo _figure_path.txt
rm -f _available_folders.txt
mkfifo _available_folders.txt

find . -maxdepth 1 -type d > _available_folders.txt &

cat _available_folders.txt | while read line; do
    LATEST_FIGURE=
    if [ -e $line/test.final/prior_samples.png ]; then
        LATEST_FIGURE=$line/test.final/prior_samples.png
    elif [ -d $line/test.snapshot ]; then
        (cd $line/test.snapshot; ls -d step_*) | sed -e "s/^step_//" | sort -r -n | while read step_idx; do
            THE_FN=$line/test.snapshot/step_"$step_idx"/prior_samples.png
            if [ -e $THE_FN ]; then
                echo $THE_FN
                break
            fi
        done > _figure_path.txt &
        LATEST_FIGURE=`cat _figure_path.txt`
    fi
    if [ -z "$LATEST_FIGURE" ]; then
        continue
    fi
    echo "$LATEST_FIGURE" "->" "$line.png"
    rm -f $line.png
    ln -s $LATEST_FIGURE $line.png
done

rm -f _figure_path.txt _available_folders.txt


