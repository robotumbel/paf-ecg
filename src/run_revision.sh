#!/usr/bin/env bash
# Revision runs for CIBM-D-26-06413 (reviewer 2, points 9 and 10).
# 1) all backbones on the official PTB-XL folds (train 1-8, val 9, test 10)
# 2) attention pooling on the same folds, paired against the CNN
# 3) the two missing xresnet1d50 seeds on the random splits
cd "$(dirname "$0")"
export PAF_DATA=data_fusion_fix TF_CPP_MIN_LOG_LEVEL=2
PAF_SPLIT=official PAF_OUT=results/paf_official PAF_SEEDS=45,46,47 \
  PAF_MODELS="CNN (ours)" ./run_gpu.sh train_paf_bench.py
PAF_SPLIT=official PAF_SEEDS=45,46,47 PAF_PREDS=results/paf_official/preds \
  ATT_OUT=results/attention_official ATT_IN_RAM=1 ./run_gpu.sh train_attention_pool.py
PAF_SPLIT=official PAF_OUT=results/paf_official PAF_SEEDS=45,46,47 \
  PAF_MODELS="CNN-LSTM,ResNet1D,Transformer" ./run_gpu.sh train_paf_bench.py
PAF_SPLIT=official PAF_OUT=results/paf_official PAF_SEEDS=45 \
  PAF_MODELS="InceptionTime,xresnet1d50" ./run_gpu.sh train_paf_bench.py
PAF_OUT=results/paf_bench_fix PAF_MODELS="xresnet1d50" ./run_gpu.sh train_paf_bench.py
echo CHAIN_DONE
