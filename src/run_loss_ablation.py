"""Fair prediction-loss ablation: KL vs RD-naive vs PL (listwise).

Same student (FreqMamba v3), same setup; only the prediction-distillation loss
differs. Each loss tuned, compared best-vs-best.

  KL (current)  : already gridded as the v3 lambda_hs=0 cells (Pred-KD only,
                  no HS). Transcribed best per dataset (no re-run).
  RD-naive (new): kd_mode rank_naive. grid lambda x K x beta.
  PL listwise   : kd_mode adaptive_rank_comp --gate_fixed 1.0 (pure PL on
                  z_ord top-K, complement off). grid lambda x K.

Usage (from seqKD/src/):
  python run_loss_ablation.py --datasets ML-1M --skip_existing
  python run_loss_ablation.py --report_only --datasets ML-1M
"""
import argparse, os, re, subprocess, sys, time

CKPT = "../../BSARec/src/output"
DS = {
    "Beauty": dict(ckpt="BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt", heads="2", alpha="0.7", c="5", drop="0.5"),
    "LastFM": dict(ckpt="BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt",  heads="1", alpha="0.9", c="3", drop="0.5"),
    "ML-1M":  dict(ckpt="BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt",  heads="1", alpha="0.3", c="9", drop="0.2"),
}
# Reference rows (transcribed).
TEACHER = {"Beauty": (0.0985, 0.0599, 0.1331, 0.0686), "LastFM": (0.0761, 0.0437, 0.1064, 0.0513),
           "ML-1M": (0.2800, 0.1572, 0.3841, 0.1835)}
NOKD = {"Beauty": (0.0910, 0.0538, 0.1270, 0.0628), "LastFM": (0.0734, 0.0389, 0.1128, 0.0488),
        "ML-1M": (0.2954, 0.1658, 0.3959, 0.1912)}
# KL pred-only best (from v3 grid lambda_hs=0 cells): (config, metrics)
KL_BEST = {
    "ML-1M":  ("lp1.0_t5.0", (0.3070, 0.1777, 0.4111, 0.2039)),
    "Beauty": ("lp1.0_t1.0", (0.0987, 0.0590, 0.1338, 0.0679)),
    "LastFM": ("lp0.5_t5.0", (0.0725, 0.0438, 0.1073, 0.0525)),
}

LAMS = [0.5, 1.0, 2.0]
KS = [10, 20, 50]
BETAS = [1.0, 1000.0]   # concentrated / ~uniform

SCORE_RE = re.compile(
    r"'HR@5':\s*'[\d.]+',\s*'NDCG@5':\s*'[\d.]+',\s*'HR@10':\s*'([\d.]+)',\s*"
    r"'NDCG@10':\s*'([\d.]+)',\s*'HR@20':\s*'([\d.]+)',\s*'NDCG@20':\s*'([\d.]+)'")


def parse(log):
    if not os.path.exists(log):
        return None
    t = open(log, encoding="utf-8", errors="replace").read()
    i = t.rfind("Test Score")
    if i < 0:
        return None
    m = SCORE_RE.findall(t[i:])
    return tuple(float(x) for x in m[-1]) if m else None


def rd_name(ds, lam, k, b):
    return f"rn_{ds}_l{lam}_k{k}_b{b}"
def pl_name(ds, lam, k):
    return f"pl_{ds}_l{lam}_k{k}"


def common_cmd(ds, name, args):
    cfg = DS[ds]
    return [sys.executable, "main.py", "--model_type", "kdstudent_v3", "--data_name", ds,
            "--train_name", name, "--alpha", cfg["alpha"], "--c", cfg["c"],
            "--d_state", "16", "--d_conv", "4", "--expand", "1", "--hidden_size", "64",
            "--num_hidden_layers", "2", "--hidden_dropout_prob", cfg["drop"],
            "--attention_probs_dropout_prob", cfg["drop"], "--lr", "0.001", "--batch_size", "256",
            "--epochs", str(args.epochs), "--patience", str(args.patience), "--seed", "42",
            "--gpu_id", args.gpu_id, "--do_distill", "--teacher_type", "bsarec",
            "--teacher_ckpt", os.path.join(CKPT, cfg["ckpt"]),
            "--teacher_num_attention_heads", cfg["heads"], "--teacher_alpha", cfg["alpha"],
            "--teacher_c", cfg["c"]]


def run_grid(ds, args):
    jobs = []
    for lam in LAMS:
        for k in KS:
            for b in BETAS:
                n = rd_name(ds, lam, k, b)
                cmd = common_cmd(ds, n, args) + ["--kd_mode", "rank_naive",
                      "--lambda_kd", str(lam), "--rank_k", str(k), "--rank_beta", str(b)]
                jobs.append((n, cmd))
    for lam in LAMS:
        for k in KS:
            n = pl_name(ds, lam, k)
            cmd = common_cmd(ds, n, args) + ["--kd_mode", "adaptive_rank_comp",
                  "--gate_fixed", "1.0", "--teacher_pre_path", "route1",
                  "--lambda_kd", str(lam), "--rank_k", str(k), "--comp_k", str(k)]
            jobs.append((n, cmd))
    total = len(jobs)
    for i, (n, cmd) in enumerate(jobs, 1):
        log = os.path.join("output", n + ".log")
        if args.skip_existing and parse(log) is not None:
            print(f"[{i}/{total}] SKIP {n}"); continue
        print(f"\n[{i}/{total}] RUN {n}")
        ts = time.perf_counter(); rc = subprocess.call(cmd)
        print(f"[{i}/{total}] DONE {n} rc={rc} {time.perf_counter()-ts:.1f}s")


def best(ds, family):
    rows = []
    if family == "rd":
        for lam in LAMS:
            for k in KS:
                for b in BETAS:
                    s = parse(os.path.join("output", rd_name(ds, lam, k, b) + ".log"))
                    if s: rows.append((f"l{lam}_k{k}_b{b}", s))
    else:
        for lam in LAMS:
            for k in KS:
                s = parse(os.path.join("output", pl_name(ds, lam, k) + ".log"))
                if s: rows.append((f"l{lam}_k{k}", s))
    if not rows: return None
    return max(rows, key=lambda r: r[1][0])  # by HR@10


def report(datasets):
    for ds in datasets:
        print(f"\n## {ds}  (prediction-loss ablation, best-of-each)\n")
        print("| Loss | best config | HR@10 | NDCG@10 | HR@20 | NDCG@20 |")
        print("|---|---|---|---|---|---|")
        print(f"| Teacher (ref) | — | " + " | ".join(f"{x:.4f}" for x in TEACHER[ds]) + " |")
        print(f"| no-KD (ref) | — | " + " | ".join(f"{x:.4f}" for x in NOKD[ds]) + " |")
        if ds in KL_BEST:
            cfg, v = KL_BEST[ds]
            print(f"| **KL** (Pred-only) | {cfg} | " + " | ".join(f"{x:.4f}" for x in v) + " |")
        rd = best(ds, "rd"); pl = best(ds, "pl")
        if rd: print(f"| **RD-naive** | {rd[0]} | " + " | ".join(f"{x:.4f}" for x in rd[1]) + " |")
        else:  print("| RD-naive | — | — | — | — | — |")
        if pl: print(f"| **PL listwise** | {pl[0]} | " + " | ".join(f"{x:.4f}" for x in pl[1]) + " |")
        else:  print("| PL listwise | — | — | — | — | — |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ML-1M"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--report_only", action="store_true")
    args = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("output", exist_ok=True)
    if not args.report_only:
        for ds in args.datasets:
            run_grid(ds, args)
    report(args.datasets)


if __name__ == "__main__":
    main()
