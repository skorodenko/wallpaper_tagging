#!/usr/bin/sh
mkdir -p ./assets/_tar
ROOT=./assets/_tar
tar --verbose -chzf $ROOT/coco.tar.gz ./assets/coco/images