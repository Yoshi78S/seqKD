"""KDStudent v2 initial run with v1's best HP per dataset.

Runs 3 experiments (Beauty / LastFM / ML-1M) with model_type=kdstudent_v2
and the hyperparameters that won the v1 grid search. The goal is to measure
the *architecture-only* effect of swapping v1 → v2 before any HP re-search.

Usage:
    python run_v2_initial.py
    python run_v2_initial.py --datasets Beauty
    python run_v2_initial.py --dry_run
    python run_v2_initial.py --skip_existing
"""
import argparse
import os
import subprocess
import sys
import time

CKPT_DIR = "../../BSARec/src/output"

# Same teacher BSARec checkpoints as KDStudent v1.
TEACHERS = {
    "Beauty": {
        "ckpt": "BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt",
        "overrides": {"teacher_num_attention_heads": "2",
                      "teacher_alpha": "0.7", "teacher_c": "5"},
    },
    "LastFM": {
        "ckpt": "BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt",
        "overrides": {"teacher_num_attention_heads": "1",
                      "teacher_alpha": "0.9", "teacher_c": "3"},
    },
    "ML-1M": {
        "ckpt": "BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt",
        "overrides": {"teacher_num_attention_heads": "1",
                      "teacher_alpha": "0.3", "teacher_c": "9"},
    },
}

# v1 grid-search winners (from KD_RESULTS.md).
BEST_HP = {
    "Beauty": {"lp": "2.0", "T": "1.0", "lhs": "0.05", "layer": "last"},
    "LastFM": {"lp": "2.0", "T": "2.0", "lhs": "0.2",  "layer": "last"},
    "ML-1M":  {"lp": "0.5", "T": "5.0", "lhs": "0.05", "layer": "last"},
}

DATASET_DROPOUT = {"Beauty": "0.5", "LastFM": "0.5", "ML-1M": "0.2"}
COMMON = {
    "model_type": "kdstudent_v2",
    "hidden_size": "64",
    "max_seq_length": "50",
    "num_hidden_layers": "2",
    "lr": "0.001",
    "batch_size": "256",
}


def build_cmd(ds, args):
    hp = BEST_HP[ds]
    ds_cfg = TEACHERS[ds]
    teacher_ckpt = os.path.join(CKPT_DIR, ds_cfg["ckpt"])
    dropout = DATASET_DROPOUT[ds]
    train_name = f"kdstudent_v2_{ds}_initial"

    cmd = [
        sys.executable, "main.py",
        "--data_name", ds,
        "--train_name", train_name,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--gpu_id", args.gpu_id,
        "--hidden_dropout_prob", dropout,
        "--attention_probs_dropout_prob", dropout,
        "--do_distill",
        "--teacher_type", "bsarec",
        "--teacher_ckpt", teacher_ckpt,
        "--lambda_kd", hp["lp"],
        "--kd_temperature", hp["T"],
        "--do_hs_distill",
        "--lambda_hs", hp["lhs"],
        "--hs_loss_type", "mse",
        "--hs_position_mode", "all",
        "--hs_layer_mode", hp["layer"],
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")

    for k, v in {**COMMON, **ds_cfg["overrides"]}.items():
        cmd.extend([f"--{k}", str(v)])

    return train_name, teacher_ckpt, cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(TEACHERS.keys()))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("output", exist_ok=True)

    print(f"[v2-initial] planned {len(args.datasets)} runs")
    t0 = time.perf_counter()
    summary = []

    for i, ds in enumerate(args.datasets, 1):
        train_name, teacher_ckpt, cmd = build_cmd(ds, args)
        log_path = os.path.join("output", f"{train_name}.log")

        if args.skip_existing and os.path.exists(log_path):
            print(f"[{i}/{len(args.datasets)}] SKIP {train_name}")
            summary.append((train_name, "skipped", 0.0))
            continue

        if not os.path.exists(teacher_ckpt):
            print(f"[{i}/{len(args.datasets)}] MISSING teacher {teacher_ckpt}")
            summary.append((train_name, "missing_teacher", 0.0))
            continue

        print(f"\n[{i}/{len(args.datasets)}] RUN {train_name}")
        if args.dry_run:
            print(f"  cmd: {' '.join(cmd)}")
            continue

        ts = time.perf_counter()
        rc = subprocess.call(cmd)
        elapsed = time.perf_counter() - ts
        status = "ok" if rc == 0 else f"fail(rc={rc})"
        print(f"[{i}/{len(args.datasets)}] DONE {train_name} {status} {elapsed:.1f}s")
        summary.append((train_name, status, elapsed))

    print(f"\n[v2-initial] total wall time: {time.perf_counter() - t0:.1f}s")
    for name, status, t in summary:
        print(f"  {name:35s} {status:18s} {t:8.1f}s")


if __name__ == "__main__":
    main()
