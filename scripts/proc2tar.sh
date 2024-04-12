#!/usr/bin/sh
mkdir -p ./assets/_tar
ROOT=./assets/_tar
tar --verbose -chzf $ROOT/proc.tar.gz ./assets/preprocessed