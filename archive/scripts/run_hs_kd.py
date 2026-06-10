"""Hidden-state KD experiment runner: BSARec teacher -> GRU4Rec student.

Usage:
    python run_hs_kd.py                     # run all experiments
    python run_hs_kd.py --phase 1           # phase 1 only (hs_only + multi_level)
    python run_hs_kd.py --phase 2           # phase 2 only (ablations)
    python run_hs_kd.py --datasets Beauty
    python run_hs_kd.py --dry_run
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

CKPT_DIR = "../../BSARec/src/output"

DATASETS = ["Beauty", "LastFM", "ML-1M"]

TEACHER_CONFIGS = {
    "Beauty": {
        "ckpt": "BSARec_Beauty_grid_lr0.0005_a0.7_c5_h4.pt",
        "overrides": {
            "teacher_num_attention_heads": "4",
            "teacher_alpha": "0.7",
            "teacher_c": "5",
        },
    },
    "LastFM": {
        "ckpt": "BSARec_LastFM_grid_lr0.001_a0.9_c9_h2.pt",
        "overrides": {
            "teacher_num_attention_heads": "2",
            "teacher_alpha": "0.9",
            "teacher_c": "9",
        },
    },
    "ML-1M": {
        "ckpt": "BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt",
        "overrides": {
            "teacher_num_attention_heads": "1",
            "teacher_alpha": "0.3",
            "teacher_c": "9",
        },
    },
}

COMMON_ARGS = {
    "hidden_size": "64",
    "max_seq_length": "50",
    "num_hidden_layers": "2",
    "hidden_dropout_prob": "0.5",
    "attention_probs_dropout_prob": "0.5",
    "gru_hidden_size": "64",
    "kd_temperature": "2.0",
}


def build_cmd(dataset, config, args):
    ds_cfg = TEACHER_CONFIGS[dataset]
    teacher_ckpt = os.path.join(CKPT_DIR, ds_cfg["ckpt"])

    cmd = [
        sys.executable, "main.py",
        "--data_name", dataset,
        "--train_name", config["train_name"],
        "--model_type", "gru4rec",
        "--teacher_type", "bsarec",
        "--teacher_ckpt", teacher_ckpt,
        "--do_hs_distill",
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--gpu_id", args.gpu_id,
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")

    all_overrides = {**COMMON_ARGS, **ds_cfg.get("overrides", {}), **config.get("overrides", {})}
    for k, v in all_overrides.items():
        cmd.extend([f"--{k}", str(v)])

    if config.get("do_distill", True):
        cmd.append("--do_distill")

    return teacher_ckpt, cmd


def phase1_configs(datasets):
    """Phase 1: HS-only and Multi-level with default settings."""
    configs = []
    for ds in datasets:
        configs.append((ds, {
            "train_name": f"gru4rec_{ds}_hs_only",
            "overrides": {"lambda_kd": "0.0", "lambda_hs": "1.0",
                          "hs_loss_type": "mse", "hs_position_mode": "all"},
            "do_distill": False,
        }))
        configs.append((ds, {
            "train_name": f"gru4rec_{ds}_multi_level",
            "overrides": {"lambda_kd": "1.0", "lambda_hs": "1.0",
                          "hs_loss_type": "mse", "hs_position_mode": "all"},
            "do_distill": True,
        }))
    return configs


def phase2_configs(datasets):
    """Phase 2: Ablation on lambda_hs, loss_type, position_mode."""
    configs = []
    for ds in datasets:
        # lambda_hs sweep (multi-level, mse, all)
        for lhs in ["0.1", "0.5", "2.0"]:
            configs.append((ds, {
                "train_name": f"gru4rec_{ds}_ml_lhs{lhs}",
                "overrides": {"lambda_kd": "1.0", "lambda_hs": lhs,
                              "hs_loss_type": "mse", "hs_position_mode": "all"},
                "do_distill": True,
            }))
        # cosine loss (multi-level, lambda_hs=1.0)
        configs.append((ds, {
            "train_name": f"gru4rec_{ds}_ml_cosine",
            "overrides": {"lambda_kd": "1.0", "lambda_hs": "1.0",
                          "hs_loss_type": "cosine", "hs_position_mode": "all"},
            "do_distill": True,
        }))
        # last position only (multi-level, lambda_hs=1.0, mse)
        configs.append((ds, {
            "train_name": f"gru4rec_{ds}_ml_last",
            "overrides": {"lambda_kd": "1.0", "lambda_hs": "1.0",
                          "hs_loss_type": "mse", "hs_position_mode": "last"},
            "do_distill": True,
        }))
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--phase", nargs="+", type=int, default=[1, 2])
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

    all_configs = []
    if 1 in args.phase:
        all_configs.extend(phase1_configs(args.datasets))
    if 2 in args.phase:
        all_configs.extend(phase2_configs(args.datasets))

    print(f"[hs-kd] planned {len(all_configs)} runs")

    summary = []
    total_start = time.perf_counter()

    for i, (dataset, config) in enumerate(all_configs, 1):
        train_name = config["train_name"]
        teacher_ckpt, cmd = build_cmd(dataset, config, args)
        log_path = os.path.join("output", f"{train_name}.log")

        if args.skip_existing and os.path.exists(log_path):
            print(f"[{i}/{len(all_configs)}] SKIP {train_name} (log exists)")
            continue

        if not os.path.exists(teacher_ckpt):
            print(f"[{i}/{len(all_configs)}] MISSING teacher {teacher_ckpt}")
            summary.append((train_name, "missing_teacher", 0.0))
            continue

        print(f"\n[{i}/{len(all_configs)}] RUN {train_name}")
        print(f"  cmd: {' '.join(cmd)}")
        if args.dry_run:
            continue

        t0 = time.perf_counter()
        rc = subprocess.call(cmd)
        elapsed = time.perf_counter() - t0
        status = "ok" if rc == 0 else f"fail(rc={rc})"
        summary.append((train_name, status, elapsed))
        print(f"[{i}/{len(all_configs)}] DONE {train_name} {status} {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_start
    print("\n========= hs-kd summary =========")
    for name, status, t in summary:
        print(f"  {name:45s} {status:18s} {t:8.1f}s")
    print(f"[hs-kd] total wall time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
