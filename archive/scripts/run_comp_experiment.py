"""Complementary-distillation ablation ladder (FreqMamba v3 student).

The ladder isolates each change with ONE variable difference:
  1. no-KD        : FreqMamba standalone (floor)                       [run]
  2. standard-KD  : KL Pred + HS (v3 grid best, HP-tuned)              [transcribed]
  3. gate1        : adaptive_rank_comp, gate_fixed=1.0  = pure PL ranking (complement OFF)  [run]
  4. gate0        : adaptive_rank_comp, gate_fixed=0.0  = PL + uniform complement           [run]
  5. adaptive     : adaptive_rank_comp, gate_fixed=-1   = PL + rho-gated complement (PROPOSED) [run]
  6. teacher      : BSARec (ceiling)                                   [transcribed]

Key comparisons:  3 vs 2 (PL vs KL+HS) | 5 vs 3 (complement effect = novelty) | 5 vs 4 (rho-gate value).
ALL run conditions use --rank_k 10 --comp_k 10 so the complement (z_ord rank ~13-17 items)
lands OUTSIDE the main PL list (pure addition, not tug-of-war).

Usage (from seqKD/src/):
  python run_comp_experiment.py --datasets ML-1M
  python run_comp_experiment.py --datasets ML-1M --skip_existing
  python run_comp_experiment.py --report_only --datasets ML-1M Beauty LastFM
"""
import argparse, os, re, subprocess, sys, time

CKPT_DIR = "../../BSARec/src/output"
DS = {
    "Beauty": dict(ckpt="BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt", heads="2", alpha="0.7", c="5", drop="0.5"),
    "LastFM": dict(ckpt="BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt",  heads="1", alpha="0.9", c="3", drop="0.5"),
    "ML-1M":  dict(ckpt="BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt",  heads="1", alpha="0.3", c="9", drop="0.2"),
}
# Transcribed (already-measured) rows.
TEACHER = {
    "Beauty": (0.0985, 0.0599, 0.1331, 0.0686),
    "LastFM": (0.0761, 0.0437, 0.1064, 0.0513),
    "ML-1M":  (0.2800, 0.1572, 0.3841, 0.1835),
}
STANDARD_KD = {  # v3 grid best (KL Pred + HS, HP-tuned)
    "Beauty": (0.0989, 0.0594, 0.1344, 0.0683),
    "LastFM": (0.0761, 0.0424, 0.1165, 0.0525),
    "ML-1M":  (0.3078, 0.1780, 0.4156, 0.2051),
}
RUN_CONDS = ["noKD", "gate1", "gate0", "adaptive"]   # conditions we actually train
SCORE_RE = re.compile(
    r"'HR@5':\s*'([\d.]+)',\s*'NDCG@5':\s*'([\d.]+)',\s*'HR@10':\s*'([\d.]+)',\s*"
    r"'NDCG@10':\s*'([\d.]+)',\s*'HR@20':\s*'([\d.]+)',\s*'NDCG@20':\s*'([\d.]+)'")


def train_name(ds, cond):
    return f"fmamba_{cond}_{ds}"


def parse_score(log):
    if not os.path.exists(log):
        return None
    txt = open(log, encoding="utf-8", errors="replace").read()
    i = txt.rfind("Test Score")
    if i < 0:
        return None
    m = SCORE_RE.findall(txt[i:])
    if not m:
        return None
    hr5, n5, hr10, n10, hr20, n20 = m[-1]
    return (float(hr10), float(n10), float(hr20), float(n20))


def build_cmd(ds, cond, args):
    cfg = DS[ds]
    common = [
        sys.executable, "main.py", "--model_type", "kdstudent_v3", "--data_name", ds,
        "--train_name", train_name(ds, cond),
        "--alpha", cfg["alpha"], "--c", cfg["c"], "--d_state", "16", "--d_conv", "4", "--expand", "1",
        "--hidden_size", "64", "--num_hidden_layers", "2",
        "--hidden_dropout_prob", cfg["drop"], "--attention_probs_dropout_prob", cfg["drop"],
        "--lr", "0.001", "--batch_size", "256", "--epochs", str(args.epochs),
        "--patience", str(args.patience), "--seed", "42", "--gpu_id", args.gpu_id,
    ]
    if cond == "noKD":
        return common  # plain Trainer (no distillation)
    # comp conditions
    gate = {"gate1": "1.0", "gate0": "0.0", "adaptive": "-1"}[cond]
    common += [
        "--do_distill", "--kd_mode", "adaptive_rank_comp",
        "--teacher_type", "bsarec", "--teacher_ckpt", os.path.join(CKPT_DIR, cfg["ckpt"]),
        "--teacher_num_attention_heads", cfg["heads"], "--teacher_alpha", cfg["alpha"], "--teacher_c", cfg["c"],
        "--teacher_pre_path", "route1", "--rank_k", "10", "--comp_k", "10",
        "--comp_beta", str(args.comp_beta), "--gate_fixed", gate,
    ]
    return common


def report(datasets):
    label = {"noKD": "Student no-KD", "standard": "+ standard-KD (KL Pred+HS)",
             "gate1": "+ gate1 (PL only, comp OFF)", "gate0": "+ gate0 (PL+uniform comp)",
             "adaptive": "+ adaptive (PROPOSED)", "teacher": "Teacher (BSARec)"}
    order = ["teacher", "noKD", "standard", "gate1", "gate0", "adaptive"]
    for ds in datasets:
        print(f"\n## {ds}\n")
        print("| Method | HR@10 | NDCG@10 | HR@20 | NDCG@20 |")
        print("|---|---|---|---|---|")
        vals = {"teacher": TEACHER[ds], "standard": STANDARD_KD[ds]}
        for cond in RUN_CONDS:
            vals[cond] = parse_score(os.path.join("output", train_name(ds, cond) + ".log"))
        for cond in order:
            v = vals.get(cond)
            row = (f"{v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[3]:.4f}"
                   if v else "— | — | — | —")
            print(f"| {label[cond]} | {row} |")
        # ladder comparisons (HR@10)
        g1, g0, ad, std = vals.get("gate1"), vals.get("gate0"), vals.get("adaptive"), vals["standard"]
        print("\n  ladder (HR@10):")
        if g1: print(f"    3 vs 2  PL-only {g1[0]:.4f} vs KL+HS {std[0]:.4f}  -> {g1[0]-std[0]:+.4f}")
        if ad and g1: print(f"    5 vs 3  adaptive {ad[0]:.4f} vs PL-only {g1[0]:.4f}  -> {ad[0]-g1[0]:+.4f}  (** complement effect **)")
        if ad and g0: print(f"    5 vs 4  adaptive {ad[0]:.4f} vs uniform {g0[0]:.4f}  -> {ad[0]-g0[0]:+.4f}  (rho-gate value)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ML-1M"])
    p.add_argument("--conditions", nargs="+", default=RUN_CONDS)
    p.add_argument("--comp_beta", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--report_only", action="store_true")
    args = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("output", exist_ok=True)

    if not args.report_only:
        combos = [(ds, c) for ds in args.datasets for c in args.conditions]
        for i, (ds, cond) in enumerate(combos, 1):
            log = os.path.join("output", train_name(ds, cond) + ".log")
            if args.skip_existing and parse_score(log) is not None:
                print(f"[{i}/{len(combos)}] SKIP {train_name(ds,cond)} (done)"); continue
            cmd = build_cmd(ds, cond, args)
            print(f"\n[{i}/{len(combos)}] RUN {train_name(ds,cond)}")
            ts = time.perf_counter()
            rc = subprocess.call(cmd)
            print(f"[{i}/{len(combos)}] DONE {train_name(ds,cond)} rc={rc} {time.perf_counter()-ts:.1f}s")

    report(args.datasets)


if __name__ == "__main__":
    main()
