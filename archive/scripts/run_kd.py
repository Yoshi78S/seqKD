"""Distillation runner: SIGMA teacher -> MLP student for each dataset.

Mirrors `BSARec/src/run_benchmarks.py` style. Reuses SIGMA teacher checkpoints
trained by the BSARec benchmark (`SIGMA_<Dataset>_bench.pt`).

Usage:
    python run_kd.py                            # run defaults on all 6 datasets
    python run_kd.py --datasets Beauty Yelp     # subset
    python run_kd.py --skip_existing            # skip when output log exists
    python run_kd.py --dry_run                  # print commands only
"""
import argparse
import os
import subprocess
import sys
import time

# Datasets must match those used in BSARec/src/run_benchmarks.py.
DEFAULT_DATASETS = [
    "LastFM", "Beauty", "ML-1M",
    "Sports_and_Outdoors", "Toys_and_Games", "Yelp",
]

# Path to BSARec's saved SIGMA teacher checkpoints (relative to seqKD/src/).
SIGMA_CKPT_DIR = "../../BSARec/src/output"

# Teacher (SIGMA) hyperparameters — must match how the SIGMA benchmark was
# trained (see BSARec/src/run_benchmarks.py README_HPARAMS for SIGMA).
TEACHER_OVERRIDES = {
    "teacher_num_hidden_layers": "1",
    "teacher_hidden_dropout_prob": "0.2",
    "teacher_attention_probs_dropout_prob": "0.2",
    # d_state/d_conv/expand are 32/4/2, which equals seqKD's parser defaults,
    # so no teacher_d_state override is needed.
}

# Student (MLP) hyperparameters — Phase 1 default settings.
STUDENT_HPARAMS = {
    "model_type": "mlp_student",
    "hidden_size": "64",
    "num_hidden_layers": "2",
    "hidden_dropout_prob": "0.5",
}

# KD-specific hyperparameters (Phase 1: simple soft-CE distillation).
KD_HPARAMS = {
    "lambda_kd": "1.0",
    "kd_temperature": "2.0",
}


def build_cmd(dataset, args):
    train_name = f"MLP_{dataset}_kd"
    teacher_ckpt = os.path.join(SIGMA_CKPT_DIR, f"SIGMA_{dataset}_bench.pt")

    cmd = [
        sys.executable, "main.py",
        "--data_name", dataset,
        "--train_name", train_name,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--gpu_id", args.gpu_id,
        "--do_distill",
        "--teacher_type", "sigma",
        "--teacher_ckpt", teacher_ckpt,
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")
    for k, v in {**STUDENT_HPARAMS, **KD_HPARAMS, **TEACHER_OVERRIDES}.items():
        cmd.extend([f"--{k}", str(v)])
    return train_name, teacher_ckpt, cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip runs whose log already exists in output/")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("output", exist_ok=True)

    print(f"[kd-runner] planned {len(args.datasets)} runs: {args.datasets}")

    summary = []
    total_start = time.perf_counter()
    for i, dataset in enumerate(args.datasets, 1):
        train_name, teacher_ckpt, cmd = build_cmd(dataset, args)
        log_path = os.path.join("output", f"{train_name}.log")

        if args.skip_existing and os.path.exists(log_path):
            print(f"[kd-runner] ({i}/{len(args.datasets)}) SKIP {train_name} (log exists)")
            continue

        if not os.path.exists(teacher_ckpt):
            print(f"[kd-runner] ({i}/{len(args.datasets)}) MISSING TEACHER "
                  f"{teacher_ckpt} -- run BSARec SIGMA benchmark first; skipping")
            summary.append((train_name, "missing_teacher", 0.0))
            continue

        print(f"\n[kd-runner] ({i}/{len(args.datasets)}) RUN {train_name}")
        print(f"[kd-runner] cmd: {' '.join(cmd)}")
        if args.dry_run:
            continue

        t0 = time.perf_counter()
        rc = subprocess.call(cmd)
        elapsed = time.perf_counter() - t0
        status = "ok" if rc == 0 else f"fail(rc={rc})"
        summary.append((train_name, status, elapsed))
        print(f"[kd-runner] ({i}/{len(args.datasets)}) DONE {train_name} {status} {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_start
    print("\n========= kd-runner summary =========")
    for name, status, t in summary:
        print(f"  {name:40s} {status:18s} {t:8.1f}s")
    print(f"[kd-runner] total wall time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
