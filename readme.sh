#!/bin/bash

# 보기 쉽게 기록해둔 readme.txt를 experiment_output_images에 복사하는 스크립트

SRC_ROOT="experiment_scripts"
DST_ROOT="experiment_output_images"

for exp_dir in "$DST_ROOT"/*; do
  exp_name=$(basename "$exp_dir")
  src_file="$SRC_ROOT/$exp_name/@readme.txt"
  dst_dir="$DST_ROOT/$exp_name"

  if [ -f "$src_file" ]; then
    cp "$src_file" "$dst_dir/"
    echo "Copied @readme.txt to $dst_dir"
  else
    echo "No @readme.txt found for $exp_name"
  fi
done
