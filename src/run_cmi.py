"""Conditional-IB corrective distillation (CEB term) sweep runner.

Runs FreqMamba-v3 + PL base + beta*I(ell;h|y) over estimators x beta, per dataset.
Each run's full training log is saved by main.py at output/<train_name>.log
(includes per-epoch 'CMI leak/CEB', the 'CMI finalize -> {...}' metrics row, and
'Test Score'). No separate CSV needed here — aggregate from the logs afterwards
(use --report_only for a quick scrape of the finalize lines).

Base distillation = PL (ranking on z_ord top-K), per-dataset PL-best HP:
  Beauty: lambda_pl=2.0 rank_k=50 | LastFM: 1.0 / 10 | ML-1M: 0.5 / 50.

Usage (from seqKD/src/):
  python run_cmi.py --datasets ML-1M                      # all estimators, ML-1M
  python run_cmi.py --datasets ML-1M --estimators linear  # Step 0 only
  python run_cmi.py --datasets ML-1M --skip_existing
  python run_cmi.py --dry_run --datasets ML-1M
  python run_cmi.py --report_only --datasets ML-1M
"""
import argparse, os, re, subprocess, sys, time

CKPT = "../../BSARec/src/output"
DS = {
    "Beauty": dict(ckpt="BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt", heads="2", alpha="0.7", c="5",
                   drop="0.5", lam_pl="2.0", rank_k="50"),
    "LastFM": dict(ckpt="BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt",  heads="1", alpha="0.9", c="3",
                   drop="0.5", lam_pl="1.0", rank_k="10"),
    "ML-1M":  dict(ckpt="BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt",  heads="1", alpha="0.3", c="9",
                   drop="0.2", lam_pl="0.5", rank_k="50"),
}
# beta sweeps per estimator (beta=0 is the G1 reference = pure PL).
EST_BETAS = {
    "linear": [0, 0.01, 0.1, 1.0],
    "adv":    [0, 0.1, 0.3, 1.0, 3.0],
    "club":   [0, 0.01, 0.05, 0.1, 0.5],
}


def train_name(ds, est, beta):
    return f"cmi_{est}_{ds}_b{beta}"


def build_cmd(ds, est, beta, args):
    cfg = DS[ds]
    return [
        sys.executable, "main.py", "--model_type", "kdstudent_v3", "--data_name", ds,
        "--train_name", train_name(ds, est, beta),
        "--do_distill", "--kd_mode", "cmi",
        "--cmi_estimator", est, "--cmi_beta", str(beta),
        "--cmi_hidden", str(args.cmi_hidden), "--cmi_critic_lr", str(args.cmi_critic_lr),
        "--cmi_critic_steps", str(args.cmi_critic_steps),
        "--teacher_type", "bsarec", "--teacher_ckpt", os.path.join(CKPT, cfg["ckpt"]),
        "--teacher_num_attention_heads", cfg["heads"], "--teacher_alpha", cfg["alpha"], "--teacher_c", cfg["c"],
        "--alpha", cfg["alpha"], "--c", cfg["c"], "--d_state", "16", "--d_conv", "4", "--expand", "1",
        "--hidden_size", "64", "--num_hidden_layers", "2",
        "--hidden_dropout_prob", cfg["drop"], "--attention_probs_dropout_prob", cfg["drop"],
        "--lr", "0.001", "--batch_size", "256", "--epochs", str(args.epochs),
        "--patience", str(args.patience), "--seed", str(args.seed), "--gpu_id", args.gpu_id,
        "--lambda_kd", cfg["lam_pl"], "--rank_k", cfg["rank_k"],
    ]


def log_done(name):
    p = os.path.join("output", name + ".log")
    if not os.path.exists(p):
        return False
    return "Test Score" in open(p, encoding="utf-8", errors="replace").read()


def report(datasets, estimators):
    """Scrape the 'CMI finalize -> ...: {dict}' line from each run's log."""
    fin = re.compile(r"CMI finalize -> .*?: (\{.*\})")
    for ds in datasets:
        print(f"\n## {ds}")
        for est in estimators:
            for beta in EST_BETAS[est]:
                name = train_name(ds, est, beta)
                p = os.path.join("output", name + ".log")
                if not os.path.exists(p):
                    print(f"  {name}: (no log)"); continue
                txt = open(p, encoding="utf-8", errors="replace").read()
                m = fin.findall(txt)
                print(f"  {est:6s} b{beta}: {m[-1] if m else '(no finalize line)'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ML-1M"])
    p.add_argument("--estimators", nargs="+", default=["linear", "adv", "club"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--cmi_hidden", type=int, default=256)
    p.add_argument("--cmi_critic_lr", type=float, default=1e-3)
    p.add_argument("--cmi_critic_steps", type=int, default=1)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--report_only", action="store_true")
    args = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("output", exist_ok=True)

    if args.report_only:
        report(args.datasets, args.estimators)
        return

    combos = [(ds, est, b) for est in args.estimators for b in EST_BETAS[est] for ds in args.datasets]
    total = len(combos)
    print(f"[cmi] planned {total} runs ({args.estimators} x betas x {args.datasets})")
    t0 = time.perf_counter()
    for i, (ds, est, beta) in enumerate(combos, 1):
        name = train_name(ds, est, beta)
        if args.skip_existing and log_done(name):
            print(f"[{i}/{total}] SKIP {name}"); continue
        cmd = build_cmd(ds, est, beta, args)
        print(f"\n[{i}/{total}] RUN {name}")
        if args.dry_run:
            print("  " + " ".join(cmd)); continue
        ts = time.perf_counter(); rc = subprocess.call(cmd)
        print(f"[{i}/{total}] DONE {name} rc={rc} {time.perf_counter()-ts:.1f}s")
    print(f"\n[cmi] total wall: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
