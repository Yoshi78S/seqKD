# seqKD — Knowledge Distillation for Sequential Recommendation

Phase 1 experiment: distill a Mamba-based teacher (SIGMA) into an MLP-based
student. Compare student-alone training vs. student-with-distillation on the
same datasets used in the BSARec benchmark.

## Layout

```
seqKD/
├── README.md
└── src/
    ├── main.py                # student training (with optional --do_distill)
    ├── dataset.py             # = BSARec/src/dataset.py
    ├── metrics.py             # = BSARec/src/metrics.py
    ├── utils.py               # arg parser extended with KD options
    ├── trainers.py            # Trainer + DistillTrainer
    ├── output/                # logs and checkpoints (created on first run)
    └── model/
        ├── __init__.py
        ├── _abstract_model.py # = BSARec/src/model/_abstract_model.py
        ├── _modules.py        # = BSARec/src/model/_modules.py
        ├── mlp_student.py     # MLP student model (new)
        └── sigma.py           # = BSARec/src/model/sigma.py (teacher)
```

Datasets and SIGMA teacher checkpoints are reused from the BSARec repo via
`--data_dir` (default `../../BSARec/src/data/`) and `--teacher_ckpt`.

## Quick start

All commands assume working directory `seqKD/src/`.

### 1. Student alone (no distillation)

```
python main.py \
  --model_type mlp_student \
  --data_name Beauty \
  --hidden_size 64 \
  --num_hidden_layers 2 \
  --hidden_dropout_prob 0.5 \
  --epochs 200 \
  --train_name MLP_Beauty_solo
```

### 2. Student + KD from SIGMA teacher

```
python main.py \
  --model_type mlp_student \
  --do_distill \
  --teacher_type sigma \
  --teacher_ckpt ../../BSARec/src/output/SIGMA_Beauty_bench.pt \
  --teacher_num_hidden_layers 1 \
  --teacher_hidden_dropout_prob 0.2 \
  --teacher_attention_probs_dropout_prob 0.2 \
  --data_name Beauty \
  --hidden_size 64 \
  --num_hidden_layers 2 \
  --hidden_dropout_prob 0.5 \
  --lambda_kd 1.0 \
  --kd_temperature 2.0 \
  --epochs 200 \
  --train_name MLP_Beauty_kd
```

The teacher's architecture-shaping flags (`--teacher_hidden_size`,
`--teacher_num_hidden_layers`, `--teacher_d_state`, ...) override the student
values when set; otherwise they inherit from the student. SIGMA Beauty
checkpoint was trained with `num_hidden_layers=1`, so override is required.

### 3. Eval a saved checkpoint

```
python main.py --do_eval --load_model MLP_Beauty_kd --data_name Beauty
```

## Loss design (Phase 1)

```
L = L_rec + lambda_kd * L_KD
```

* `L_rec` — student cross-entropy over the full item vocabulary, computed at
  the last sequence position. Identical to the recommendation loss SIGMA
  uses, so the comparison is apples-to-apples.
* `L_KD` — `KL(softmax(teacher_logits/T) || log_softmax(student_logits/T)) * T^2`,
  the SimRec-style soft-label cross-entropy with temperature scaling.

Phase 2/3 (sequence-length-adaptive weighting, teacher-confidence-based
scaling) are not implemented yet.
