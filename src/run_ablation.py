"""Tier-1 ablation runner for KDStudent.

Two ablation families per dataset:

1. ARCHITECTURE (6 runs per dataset)
   Reference (full KDStudent) plus 5 single-component removals:
     - no_pos_emb    : drop position embeddings
     - no_input_ln   : drop input LayerNorm + dropout
     - no_ffn        : drop block-internal FFN
     - no_block_ln   : drop block-internal LayerNorm
     - flat_gru      : replace block stack with one nn.GRU(num_layers=N).
                       HS-KD is dropped here because per-block hooks no longer
                       have targets; this run is Pred-KD only.
   All five removals run with the dataset's best HP from the KDStudent grid.

2. DISTILLATION (4 runs per dataset)
   Decompose the loss term-by-term, on the FULL KDStudent architecture:
     - d1_standalone : L_rec only         (no --do_distill)
     - d2_pred_only  : L_rec + L_pred     (λ_pred=best, λ_hs=0)
     - d3_hs_only    : L_rec + L_hs       (λ_pred=0, λ_hs=best)
     - d4_full       : L_rec + L_pred + L_hs   (full best HP)

Total per dataset: 6 + 4 = 10 runs, but the "ref" arch run and "d4_full" share
the same config, so the runner runs the unique 9. Across 3 datasets: 27 runs.

Usage:
    python run_ablation.py
    python run_ablation.py --datasets Beauty
    python run_ablation.py --families arch
    python run_ablation.py --skip_existing
    python run_ablation.py --dry_run
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

CKPT_DIR = "../../BSARec/src/output"
SEQKD_OUT = "output"

# Best HP per dataset (from KD_RESULTS.md → KDStudent Best Hyperparameters).
BEST_HP = {
    "Beauty": {"lp": "2.0", "T": "1.0", "lhs": "0.05", "layer": "last"},
    "LastFM": {"lp": "2.0", "T": "2.0", "lhs": "0.2",  "layer": "last"},
    "ML-1M":  {"lp": "0.5", "T": "5.0", "lhs": "0.05", "layer": "last"},
}

# BSARec teacher per dataset.
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

DATASET_DROPOUT = {"Beauty": "0.5", "LastFM": "0.5", "ML-1M": "0.2"}
COMMON = {
    "model_type": "kdstudent",
    "hidden_size": "64",
    "max_seq_length": "50",
    "num_hidden_layers": "2",
    "lr": "0.001",
    "batch_size": "256",
}

ARCH_ABLATIONS = [
    # The full-architecture reference is dist/d4_full (same config), so we skip
    # a redundant "ref" run here. Architecture-removal variants only.
    ("no_pos_emb",   ["--abl_no_pos_emb"]),
    ("no_input_ln",  ["--abl_no_input_ln"]),
    ("no_ffn",       ["--abl_no_ffn"]),
    ("no_block_ln",  ["--abl_no_block_ln"]),
    ("flat_gru",     ["--abl_flat_gru"]),
]

DIST_STAGES = ["d1_standalone", "d2_pred_only", "d3_hs_only", "d4_full"]


def base_cmd(ds, train_name, args):
    ds_cfg = TEACHERS[ds]
    dropout = DATASET_DROPOUT[ds]
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
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")
    return cmd, ds_cfg


def attach_teacher(cmd, ds_cfg):
    cmd.extend([
        "--teacher_type", "bsarec",
        "--teacher_ckpt", os.path.join(CKPT_DIR, ds_cfg["ckpt"]),
    ])
    for k, v in ds_cfg["overrides"].items():
        cmd.extend([f"--{k}", str(v)])


def attach_common(cmd):
    for k, v in COMMON.items():
        cmd.extend([f"--{k}", str(v)])


# ── Builders ──────────────────────────────────────────────────

def build_arch(ds, variant, extra_flags, args):
    hp = BEST_HP[ds]
    train_name = f"kdstudent_{ds}_abl_arch_{variant}"
    cmd, ds_cfg = base_cmd(ds, train_name, args)
    cmd.extend(["--do_distill",
                "--lambda_kd", hp["lp"],
                "--kd_temperature", hp["T"]])
    # flat_gru drops HS-KD (no per-block hooks exist)
    if variant != "flat_gru":
        cmd.extend(["--do_hs_distill",
                    "--lambda_hs", hp["lhs"],
                    "--hs_loss_type", "mse",
                    "--hs_position_mode", "all",
                    "--hs_layer_mode", hp["layer"]])
    attach_teacher(cmd, ds_cfg)
    attach_common(cmd)
    cmd.extend(extra_flags)
    return train_name, cmd


def build_dist(ds, stage, args):
    hp = BEST_HP[ds]
    train_name = f"kdstudent_{ds}_abl_dist_{stage}"
    cmd, ds_cfg = base_cmd(ds, train_name, args)

    if stage == "d1_standalone":
        # plain Trainer, no teacher needed
        attach_common(cmd)
        return train_name, cmd

    # All other stages require teacher
    if stage == "d2_pred_only":
        cmd.extend(["--do_distill",
                    "--lambda_kd", hp["lp"],
                    "--kd_temperature", hp["T"]])
    elif stage == "d3_hs_only":
        # Pred-KD weight set to 0 → loss is L_rec + λ_hs · L_hs
        cmd.extend(["--do_distill",
                    "--lambda_kd", "0.0",
                    "--kd_temperature", hp["T"],
                    "--do_hs_distill",
                    "--lambda_hs", hp["lhs"],
                    "--hs_loss_type", "mse",
                    "--hs_position_mode", "all",
                    "--hs_layer_mode", hp["layer"]])
    elif stage == "d4_full":
        cmd.extend(["--do_distill",
                    "--lambda_kd", hp["lp"],
                    "--kd_temperature", hp["T"],
                    "--do_hs_distill",
                    "--lambda_hs", hp["lhs"],
                    "--hs_loss_type", "mse",
                    "--hs_position_mode", "all",
                    "--hs_layer_mode", hp["layer"]])
    attach_teacher(cmd, ds_cfg)
    attach_common(cmd)
    return train_name, cmd


# ── Runner ────────────────────────────────────────────────────

def run_one(train_name, cmd, args):
    log_path = os.path.join(SEQKD_OUT, f"{train_name}.log")
    if args.skip_existing and os.path.exists(log_path):
        print(f"  SKIP {train_name}")
        return "skipped", 0.0
    print(f"  RUN  {train_name}")
    if args.dry_run:
        print(f"    cmd: {' '.join(cmd)}")
        return "dry", 0.0
    t0 = time.perf_counter()
    rc = subprocess.call(cmd)
    elapsed = time.perf_counter() - t0
    status = "ok" if rc == 0 else f"fail(rc={rc})"
    print(f"  DONE {train_name} {status} {elapsed:.1f}s")
    return status, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(TEACHERS.keys()))
    p.add_argument("--families", nargs="+", default=["arch", "dist"],
                   choices=["arch", "dist"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(SEQKD_OUT, exist_ok=True)

    runs = []
    if "arch" in args.families:
        for ds in args.datasets:
            for variant, flags in ARCH_ABLATIONS:
                name, cmd = build_arch(ds, variant, flags, args)
                runs.append((ds, "arch", variant, name, cmd))
    if "dist" in args.families:
        for ds in args.datasets:
            for stage in DIST_STAGES:
                name, cmd = build_dist(ds, stage, args)
                runs.append((ds, "dist", stage, name, cmd))

    print(f"[ablation] planned {len(runs)} runs")
    t0 = time.perf_counter()
    for i, (ds, family, variant, name, cmd) in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {ds} / {family} / {variant}")
        run_one(name, cmd, args)
    print(f"\n[ablation] total wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
