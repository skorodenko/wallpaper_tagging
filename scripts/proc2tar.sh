#!/usr/bin/sh
mkdir -p ./assets/_tar
ROOT=./assets/_tar
tar --verbose -chzf $ROOT/proc.tar.gz ./assets/preprocessed ./assets/coco/annotations \
    ./assets/nus-wide/TagList1k.txt ./assets/nus-wide/TagList81.txt
