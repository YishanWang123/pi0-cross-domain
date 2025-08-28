#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate pi0
# 或者用 uv，但不要两者一起用
# export XLA_PYTHON_CLIENT_MEM_FRACTION 可保留
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export CUDA_VISIBLE_DEVICES=0,1,2,3

python scripts/train.py pi0_cotrain_libero_robomimic --exp-name=test --data.assets.asset-id=cotrain --overwrite 2>&1 | tee train.log