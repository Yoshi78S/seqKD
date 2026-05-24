"""Full KDStudent grid search: 4 × 3 × 7 = 84 runs per dataset.

Search axes per (dataset):
    λ_pred  ∈ {0.1, 0.5, 1.0, 2.0}
    T       ∈ {1.0, 2.0, 5.0}
    (λ_hs, layer_mode) ∈ {
        (0,    none),                      # Pred-KD only
        (0.05, last), (0.05, all),
        (0.1,  last), (0.1,  all),
        (0.2,  last), (0.2,  all),
    }

3 datasets × 84 = **252 runs**.

Fixed: lr=0.001, batch=256, patience=10, MSE, position_mode='all'.
Dropout: Beauty/LastFM = 0.5, ML-1M = 0.2.
Teacher: BSARec best checkpoint per dataset (new grid search).

Usage:
    python run_kdstudent_grid.py                    # run all 252
    python run_kdstudent_grid.py --datasets Beauty  # subset
    python run_kdstudent_grid.py --skip_existing
    python run_kdstudent_grid.py --dry_run
    python run_kdstudent_grid.py --report_only      # just print tables from existing logs
"""
import argparse
import itertools
import os
import re
import subprocess
import sys
import time

CKPT_DIR = "../../BSARec/src/output"

# Per-dataset BSARec teacher checkpoint (best HR@10 from grid search).
# Hardcoded here after grid completion — see `aggregate_baseline_grid.py`
# for how the best HPs are selected.
TEACHERS = {
    "Beauty": {
        "ckpt": "BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt",
        "overrides": {
            "teacher_num_attention_heads": "2",
            "teacher_alpha": "0.7",
            "teacher_c": "5",
        },
    },
    "LastFM": {
        "ckpt": "BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt",
        "overrides": {
            "teacher_num_attention_heads": "1",
            "teacher_alpha": "0.9",
            "teacher_c": "3",
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

DATASET_DROPOUT = {"Beauty": "0.5", "LastFM": "0.5", "ML-1M": "0.2"}

COMMON = {
    "model_type": "kdstudent",
    "hidden_size": "64",
    "max_seq_length": "50",
    "num_hidden_layers": "2",
    "lr": "0.001",
    "batch_size": "256",
}

# ── Grid axes ────────────────────────────────────────────────
LP_GRID = [0.1, 0.5, 1.0, 2.0]
T_GRID = [1.0, 2.0, 5.0]
HS_GRID = [
    (0.0, None),
    (0.05, "last"), (0.05, "all"),
    (0.1,  "last"), (0.1,  "all"),
    (0.2,  "last"), (0.2,  "all"),
]

SCORE_RE = re.compile(
    r"'Epoch':\s*\d+,\s*"
    r"'HR@5':\s*'([\d.]+)',\s*'NDCG@5':\s*'([\d.]+)',\s*"
    r"'HR@10':\s*'([\d.]+)',\s*'NDCG@10':\s*'([\d.]+)',\s*"
    r"'HR@20':\s*'([\d.]+)',\s*'NDCG@20':\s*'([\d.]+)'"
)


def train_name_for(ds, lp, T, lhs, layer):
    if lhs == 0.0:
        return f"kdstudent_{ds}_grid_lp{lp}_t{T}_lhs0"
    return f"kdstudent_{ds}_grid_lp{lp}_t{T}_lhs{lhs}_{layer}"


def parse_test_score(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    idx = text.rfind("Test Score")
    if idx < 0:
        return None
    m = SCORE_RE.findall(text[idx:])
    if not m:
        return None
    hr5, ndcg5, hr10, ndcg10, hr20, ndcg20 = m[-1]
    return {
        "HR@5": float(hr5), "NDCG@5": float(ndcg5),
        "HR@10": float(hr10), "NDCG@10": float(ndcg10),
        "HR@20": float(hr20), "NDCG@20": float(ndcg20),
    }


def build_cmd(ds, lp, T, lhs, layer, args):
    train_name = train_name_for(ds, lp, T, lhs, layer)
    ds_cfg = TEACHERS[ds]
    teacher_ckpt = os.path.join(CKPT_DIR, ds_cfg["ckpt"])
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
        "--do_distill",
        "--teacher_type", "bsarec",
        "--teacher_ckpt", teacher_ckpt,
        "--lambda_kd", str(lp),
        "--kd_temperature", str(T),
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")

    if lhs > 0:
        cmd.extend([
            "--do_hs_distill",
            "--lambda_hs", str(lhs),
            "--hs_loss_type", "mse",
            "--hs_position_mode", "all",
            "--hs_layer_mode", layer,
        ])

    overrides = {**COMMON, **ds_cfg["overrides"]}
    for k, v in overrides.items():
        cmd.extend([f"--{k}", str(v)])

    return train_name, teacher_ckpt, cmd


# ── Reporting ────────────────────────────────────────────────

def print_dataset_report(ds, results):
    """One flat table per dataset, sorted by HR@10 descending."""
    print(f"\n## {ds}\n")
    print("| λ_pred | T | λ_hs | layer | HR@10 | NDCG@10 |")
    print("|---|---|---|---|---|---|")

    rows = []
    for (lp, T, lhs, layer), score in results.items():
        if score is None:
            continue
        layer_str = "—" if lhs == 0.0 else layer
        rows.append((score["HR@10"], score["NDCG@10"], lp, T, lhs, layer_str))

    if not rows:
        print(f"| — | — | — | — | — | — |  (no completed runs)")
        return None

    rows.sort(key=lambda r: r[0], reverse=True)
    for hr, ndcg, lp, T, lhs, layer_str in rows:
        print(f"| {lp} | {T} | {lhs} | {layer_str} | {hr:.4f} | {ndcg:.4f} |")

    hr_best, ndcg_best, lp, T, lhs, layer_str = rows[0]
    return ((lp, T, lhs, layer_str), {"HR@10": hr_best, "NDCG@10": ndcg_best})


def print_overall_summary(best_per_ds):
    print("\n## Best per Dataset\n")
    print("| Dataset | λ_pred | T | λ_hs | layer | HR@10 | NDCG@10 |")
    print("|---|---|---|---|---|---|---|")
    for ds, item in best_per_ds.items():
        if item is None:
            print(f"| {ds} | — | — | — | — | — | — |")
            continue
        (lp, T, lhs, layer_str), score = item
        print(f"| {ds} | {lp} | {T} | {lhs} | {layer_str} "
              f"| {score['HR@10']:.4f} | {score['NDCG@10']:.4f} |")


def collect_results(ds):
    results = {}
    for lp in LP_GRID:
        for T in T_GRID:
            for lhs, layer in HS_GRID:
                name = train_name_for(ds, lp, T, lhs, layer)
                log_path = os.path.join("output", f"{name}.log")
                results[(lp, T, lhs, layer)] = parse_test_score(log_path)
    return results


# ── Main loop ────────────────────────────────────────────────

def run_grid(args):
    total_combos = list(itertools.product(args.datasets, LP_GRID, T_GRID, HS_GRID))
    total = len(total_combos)
    print(f"[grid] planned {total} runs across {len(args.datasets)} datasets")

    summary = []
    t0 = time.perf_counter()

    for i, (ds, lp, T, (lhs, layer)) in enumerate(total_combos, 1):
        train_name, teacher_ckpt, cmd = build_cmd(ds, lp, T, lhs, layer, args)
        log_path = os.path.join("output", f"{train_name}.log")

        if args.skip_existing and os.path.exists(log_path):
            print(f"[{i}/{total}] SKIP {train_name}")
            continue

        if not os.path.exists(teacher_ckpt):
            print(f"[{i}/{total}] MISSING teacher {teacher_ckpt}")
            summary.append((train_name, "missing_teacher", 0.0))
            continue

        print(f"\n[{i}/{total}] RUN {train_name}")
        if args.dry_run:
            print(f"  cmd: {' '.join(cmd)}")
            continue

        ts = time.perf_counter()
        rc = subprocess.call(cmd)
        elapsed = time.perf_counter() - ts
        status = "ok" if rc == 0 else f"fail(rc={rc})"
        summary.append((train_name, status, elapsed))
        print(f"[{i}/{total}] DONE {train_name} {status} {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - t0
    print(f"\n[grid] total wall time: {total_elapsed:.1f}s")


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
    p.add_argument("--report_only", action="store_true",
                   help="Skip training; just parse existing logs and print tables.")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("output", exist_ok=True)

    if not args.report_only:
        run_grid(args)

    # Report
    best_per_ds = {}
    for ds in args.datasets:
        results = collect_results(ds)
        best_per_ds[ds] = print_dataset_report(ds, results)
    print_overall_summary(best_per_ds)


if __name__ == "__main__":
    main()
