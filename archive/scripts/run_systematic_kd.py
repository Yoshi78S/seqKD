"""Systematic KD runner: 3 teachers x 4 students x 3 datasets.

Usage:
    python run_systematic_kd.py                          # run all 36 combinations
    python run_systematic_kd.py --skip_existing          # skip if log exists
    python run_systematic_kd.py --dry_run                # print commands only
    python run_systematic_kd.py --students gru4rec lrurec --teachers bsarec
    python run_systematic_kd.py --datasets Beauty
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

CKPT_DIR = "../../BSARec/src/output"

DATASETS = ["Beauty", "LastFM", "ML-1M"]
STUDENTS = ["mlp_student", "gru4rec", "lrurec", "fmlprec"]

STUDENT_HPARAMS = {
    "hidden_size": "64",
    "max_seq_length": "50",
    "num_hidden_layers": "2",
    "hidden_dropout_prob": "0.5",
    "attention_probs_dropout_prob": "0.5",
}

KD_HPARAMS = {
    "lambda_kd": "1.0",
    "kd_temperature": "2.0",
}

TEACHERS = {
    "sigma": {
        "Beauty": {
            "ckpt": "SIGMA_Beauty_grid_ds32_l2.pt",
            "overrides": {
                "teacher_num_hidden_layers": "2",
                "teacher_hidden_dropout_prob": "0.2",
                "teacher_attention_probs_dropout_prob": "0.2",
                "teacher_d_state": "32",
            },
        },
        "LastFM": {
            "ckpt": "SIGMA_LastFM_grid_ds8_l4.pt",
            "overrides": {
                "teacher_num_hidden_layers": "4",
                "teacher_hidden_dropout_prob": "0.2",
                "teacher_attention_probs_dropout_prob": "0.2",
                "teacher_d_state": "8",
            },
        },
        "ML-1M": {
            "ckpt": "SIGMA_ML-1M_grid_ds16_l4.pt",
            "overrides": {
                "teacher_num_hidden_layers": "4",
                "teacher_hidden_dropout_prob": "0.2",
                "teacher_attention_probs_dropout_prob": "0.2",
                "teacher_d_state": "16",
            },
        },
    },
    "bsarec": {
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
    },
    "duorec": {
        "Beauty": {
            "ckpt": "DuoRec_Beauty_grid_d0.5_tau1.0.pt",
            "overrides": {
                "teacher_tau": "1.0",
                "teacher_lmd": "0.1",
                "teacher_lmd_sem": "0.1",
                "teacher_ssl": "us_x",
                "teacher_sim": "dot",
            },
        },
        "LastFM": {
            "ckpt": "DuoRec_LastFM_grid_d0.5_tau1.0.pt",
            "overrides": {
                "teacher_tau": "1.0",
                "teacher_lmd": "0.1",
                "teacher_lmd_sem": "0.1",
                "teacher_ssl": "us_x",
                "teacher_sim": "dot",
            },
        },
        "ML-1M": {
            "ckpt": "DuoRec_ML-1M_grid_d0.2_tau0.1.pt",
            "overrides": {
                "teacher_tau": "0.1",
                "teacher_lmd": "0.1",
                "teacher_lmd_sem": "0.1",
                "teacher_ssl": "us_x",
                "teacher_sim": "dot",
            },
        },
    },
}


def build_cmd(teacher_type, student_type, dataset, args):
    train_name = f"{student_type}_{dataset}_kd_{teacher_type}_v2"

    ds_cfg = TEACHERS[teacher_type][dataset]
    teacher_ckpt = os.path.join(CKPT_DIR, ds_cfg["ckpt"])
    overrides = ds_cfg.get("overrides", {})

    cmd = [
        sys.executable, "main.py",
        "--data_name", dataset,
        "--train_name", train_name,
        "--model_type", student_type,
        "--teacher_type", teacher_type,
        "--teacher_ckpt", teacher_ckpt,
        "--do_distill",
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--gpu_id", args.gpu_id,
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")

    for k, v in {**STUDENT_HPARAMS, **KD_HPARAMS, **overrides}.items():
        cmd.extend([f"--{k}", str(v)])

    if student_type == "gru4rec":
        cmd.extend(["--gru_hidden_size", "64"])

    return train_name, teacher_ckpt, cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", nargs="+", default=list(TEACHERS.keys()))
    p.add_argument("--students", nargs="+", default=STUDENTS)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
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

    combos = list(itertools.product(args.teachers, args.students, args.datasets))
    print(f"[systematic-kd] planned {len(combos)} runs")

    summary = []
    total_start = time.perf_counter()

    for i, (teacher, student, dataset) in enumerate(combos, 1):
        train_name, teacher_ckpt, cmd = build_cmd(teacher, student, dataset, args)
        log_path = os.path.join("output", f"{train_name}.log")

        if args.skip_existing and os.path.exists(log_path):
            print(f"[{i}/{len(combos)}] SKIP {train_name} (log exists)")
            continue

        if not os.path.exists(teacher_ckpt):
            print(f"[{i}/{len(combos)}] MISSING teacher {teacher_ckpt} -- skipping")
            summary.append((train_name, "missing_teacher", 0.0))
            continue

        print(f"\n[{i}/{len(combos)}] RUN {train_name}")
        print(f"  cmd: {' '.join(cmd)}")
        if args.dry_run:
            continue

        t0 = time.perf_counter()
        rc = subprocess.call(cmd)
        elapsed = time.perf_counter() - t0
        status = "ok" if rc == 0 else f"fail(rc={rc})"
        summary.append((train_name, status, elapsed))
        print(f"[{i}/{len(combos)}] DONE {train_name} {status} {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_start
    print("\n========= systematic-kd summary =========")
    for name, status, t in summary:
        print(f"  {name:45s} {status:18s} {t:8.1f}s")
    print(f"[systematic-kd] total wall time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
