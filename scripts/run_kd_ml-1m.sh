#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../src"

python main.py \
  --model_type mlp_student --do_distill \
  --teacher_type sigma \
  --teacher_ckpt ../../BSARec/src/output/SIGMA_ML-1M_bench.pt \
  --teacher_num_hidden_layers 1 \
  --teacher_hidden_dropout_prob 0.2 \
  --teacher_attention_probs_dropout_prob 0.2 \
  --data_name ML-1M \
  --hidden_size 64 --num_hidden_layers 2 \
  --hidden_dropout_prob 0.5 \
  --lambda_kd 1.0 --kd_temperature 2.0 \
  --epochs 200 \
  --train_name MLP_ML-1M_kd
