"""KDStudent v3 (FreqMamba) grid search / experiment runner.

FreqMamba = BSARec's two-branch fusion with the attention branch replaced by a
residual-free Mamba (see model/kd_student_v3.py). The frequency branch is
BSARec-identical, so the student's fusion weight `alpha` and cutoff `c` default
to the *teacher's* per-dataset values (structural inheritance).

Three modes (`--mode`):
  quick    : 1 run/DS — the v1-winner KD config at teacher alpha. End-to-end
             sanity check before committing to a big grid. (3 runs total)
  focused  : 3×3×4 = 36 KD configs/DS (default). Mirrors v1's grid but drops
             λ_pred=0.1 and the layer='all' HS mode (both rarely won in v1).
             (108 runs total)
  alpha    : fix KD to the v1-winner config, sweep fusion alpha ∈
             {0.1,0.3,0.5,0.7,0.9}. Tunes the freq∥Mamba balance. (15 runs)

Mamba is fixed to expand=1, d_state=16, d_conv=4 (lightweight) — override with
--expand/--d_state/--d_conv.

Usage (from seqKD/src/):
  python run_kdstudent_v3_grid.py --mode quick                 # validate first
  python run_kdstudent_v3_grid.py --mode quick --epochs 2      # ultra-fast smoke
  python run_kdstudent_v3_grid.py --datasets Beauty            # focused, one DS
  python run_kdstudent_v3_grid.py --mode alpha --datasets LastFM
  python run_kdstudent_v3_grid.py --skip_existing              # resume
  python run_kdstudent_v3_grid.py --report_only                # tables from logs
  python run_kdstudent_v3_grid.py --dry_run                    # print commands
"""
import argparse
import itertools
import os
import re
import subprocess
import sys
import time

CKPT_DIR = "../../BSARec/src/output"

# Per-dataset config. `alpha` / `c` are the BSARec teacher's values, inherited
# by the student's frequency branch. Teacher is the best BSARec checkpoint.
DATASETS = {
    "Beauty": {
        "ckpt": "BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt",
        "teacher_heads": "2", "alpha": "0.7", "c": "5", "dropout": "0.5",
        "v1_best": dict(lp=2.0, T=1.0, lhs=0.05, layer="last"),
    },
    "LastFM": {
        "ckpt": "BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt",
        "teacher_heads": "1", "alpha": "0.9", "c": "3", "dropout": "0.5",
        "v1_best": dict(lp=2.0, T=2.0, lhs=0.2, layer="last"),
    },
    "ML-1M": {
        "ckpt": "BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt",
        "teacher_heads": "1", "alpha": "0.3", "c": "9", "dropout": "0.2",
        "v1_best": dict(lp=0.5, T=5.0, lhs=0.05, layer="last"),
    },
}

COMMON = {
    "model_type": "kdstudent_v3",
    "hidden_size": "64",
    "max_seq_length": "50",
    "num_hidden_layers": "2",
    "lr": "0.001",
    "batch_size": "256",
}

# ── Grid axes (focused mode) ─────────────────────────────────
LP_GRID = [0.5, 1.0, 2.0]
T_GRID = [1.0, 2.0, 5.0]
HS_GRID = [
    (0.0, None),
    (0.05, "last"),
    (0.1, "last"),
    (0.2, "last"),
]
ALPHA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]   # alpha mode only

SCORE_RE = re.compile(
    r"'Epoch':\s*\d+,\s*"
    r"'HR@5':\s*'([\d.]+)',\s*'NDCG@5':\s*'([\d.]+)',\s*"
    r"'HR@10':\s*'([\d.]+)',\s*'NDCG@10':\s*'([\d.]+)',\s*"
    r"'HR@20':\s*'([\d.]+)',\s*'NDCG@20':\s*'([\d.]+)'"
)


def train_name_for(ds, lp, T, lhs, layer, alpha=None):
    a = "" if alpha is None else f"_a{alpha}"
    if lhs == 0.0:
        return f"kdstudent_v3_{ds}{a}_lp{lp}_t{T}_lhs0"
    return f"kdstudent_v3_{ds}{a}_lp{lp}_t{T}_lhs{lhs}_{layer}"


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


def configs_for_mode(ds, mode):
    """Yield (lp, T, lhs, layer, alpha) tuples for the given mode.

    alpha is None when the dataset-default (teacher) alpha is used; otherwise it
    is the swept value (and goes into the train_name).
    """
    cfg = DATASETS[ds]
    if mode == "quick":
        b = cfg["v1_best"]
        yield (b["lp"], b["T"], b["lhs"], b["layer"], None)
    elif mode == "alpha":
        b = cfg["v1_best"]
        for a in ALPHA_GRID:
            yield (b["lp"], b["T"], b["lhs"], b["layer"], a)
    else:  # focused
        for lp in LP_GRID:
            for T in T_GRID:
                for lhs, layer in HS_GRID:
                    yield (lp, T, lhs, layer, None)


def build_cmd(ds, lp, T, lhs, layer, alpha, args):
    cfg = DATASETS[ds]
    teacher_ckpt = os.path.join(CKPT_DIR, cfg["ckpt"])
    # student fusion alpha: swept value, else teacher's (inherited)
    student_alpha = cfg["alpha"] if alpha is None else str(alpha)
    train_name = train_name_for(ds, lp, T, lhs, layer, alpha)

    cmd = [
        sys.executable, "main.py",
        "--data_name", ds,
        "--train_name", train_name,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--gpu_id", args.gpu_id,
        "--hidden_dropout_prob", cfg["dropout"],
        "--attention_probs_dropout_prob", cfg["dropout"],
        # FreqMamba student knobs
        "--alpha", student_alpha,         # fusion weight (freq branch)
        "--c", cfg["c"],                  # freq cutoff (inherited)
        "--d_state", str(args.d_state),
        "--d_conv", str(args.d_conv),
        "--expand", str(args.expand),
        # distillation
        "--do_distill",
        "--teacher_type", "bsarec",
        "--teacher_ckpt", teacher_ckpt,
        "--lambda_kd", str(lp),
        "--kd_temperature", str(T),
        # teacher arch overrides (build BSARec teacher correctly)
        "--teacher_num_attention_heads", cfg["teacher_heads"],
        "--teacher_alpha", cfg["alpha"],
        "--teacher_c", cfg["c"],
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
    for k, v in COMMON.items():
        cmd.extend([f"--{k}", str(v)])

    return train_name, teacher_ckpt, cmd


# ── Reporting ────────────────────────────────────────────────

def print_dataset_report(ds, mode):
    print(f"\n## {ds} ({mode})\n")
    print("| alpha | λ_pred | T | λ_hs | layer | HR@10 | NDCG@10 |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    cfg = DATASETS[ds]
    for (lp, T, lhs, layer, alpha) in configs_for_mode(ds, mode):
        name = train_name_for(ds, lp, T, lhs, layer, alpha)
        score = parse_test_score(os.path.join("output", f"{name}.log"))
        if score is None:
            continue
        a_str = cfg["alpha"] if alpha is None else str(alpha)
        layer_str = "—" if lhs == 0.0 else layer
        rows.append((score["HR@10"], score["NDCG@10"], a_str, lp, T, lhs, layer_str))
    if not rows:
        print("| — | — | — | — | — | — | — |  (no completed runs)")
        return None
    rows.sort(key=lambda r: r[0], reverse=True)
    for hr, ndcg, a_str, lp, T, lhs, layer_str in rows:
        print(f"| {a_str} | {lp} | {T} | {lhs} | {layer_str} | {hr:.4f} | {ndcg:.4f} |")
    hr, ndcg, a_str, lp, T, lhs, layer_str = rows[0]
    return {"alpha": a_str, "lp": lp, "T": T, "lhs": lhs, "layer": layer_str,
            "HR@10": hr, "NDCG@10": ndcg}


def print_overall_summary(best_per_ds):
    print("\n## Best per Dataset\n")
    print("| Dataset | alpha | λ_pred | T | λ_hs | layer | HR@10 | NDCG@10 |")
    print("|---|---|---|---|---|---|---|---|")
    for ds, b in best_per_ds.items():
        if b is None:
            print(f"| {ds} | — | — | — | — | — | — | — |")
            continue
        print(f"| {ds} | {b['alpha']} | {b['lp']} | {b['T']} | {b['lhs']} "
              f"| {b['layer']} | {b['HR@10']:.4f} | {b['NDCG@10']:.4f} |")


# ── Main loop ────────────────────────────────────────────────

def run_grid(args):
    combos = [(ds, *c) for ds in args.datasets
              for c in configs_for_mode(ds, args.mode)]
    total = len(combos)
    print(f"[v3-grid] mode={args.mode}, planned {total} runs "
          f"across {len(args.datasets)} datasets "
          f"(Mamba expand={args.expand}, d_state={args.d_state})")

    t0 = time.perf_counter()
    for i, (ds, lp, T, lhs, layer, alpha) in enumerate(combos, 1):
        train_name, teacher_ckpt, cmd = build_cmd(ds, lp, T, lhs, layer, alpha, args)
        log_path = os.path.join("output", f"{train_name}.log")

        if args.skip_existing and os.path.exists(log_path):
            print(f"[{i}/{total}] SKIP {train_name}")
            continue
        if not os.path.exists(teacher_ckpt):
            print(f"[{i}/{total}] MISSING teacher {teacher_ckpt}")
            continue
        print(f"\n[{i}/{total}] RUN {train_name}")
        if args.dry_run:
            print(f"  cmd: {' '.join(cmd)}")
            continue
        ts = time.perf_counter()
        rc = subprocess.call(cmd)
        elapsed = time.perf_counter() - ts
        status = "ok" if rc == 0 else f"fail(rc={rc})"
        print(f"[{i}/{total}] DONE {train_name} {status} {elapsed:.1f}s")

    print(f"\n[v3-grid] total wall time: {time.perf_counter() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "focused", "alpha"],
                   default="focused")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=1)
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--report_only", action="store_true")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("output", exist_ok=True)

    if not args.report_only:
        run_grid(args)

    best_per_ds = {}
    for ds in args.datasets:
        best_per_ds[ds] = print_dataset_report(ds, args.mode)
    print_overall_summary(best_per_ds)


if __name__ == "__main__":
    main()
