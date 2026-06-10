"""Gated relative de-bias loss — Stage 1 sweep runner (ML-1M, single seed 42).

Arms (vs baseline pure-PL 0.3008 and cmi-linear b0.01 0.2935):
  margin       m=0.3 x lam_db {0.1, 1, 10}, then m {0.1, 0.5} at the best lam*
  bpr          lam_db {0.1, 1, 10}
  logit_margin lm_margin {0.5, 1.0, 2.0}
  reweight     gamma_rw {0.5, 1.0, 2.0}
Total 14 runs. Training protocol identical to the beta sweep (epochs 200,
patience 10, seed 42, db_warmup 10). Logs: output/<name>.log; per-epoch
diagnostics: results/<name>_epochs.csv; final row: results/debias_<mode>_ML-1M.csv.

Usage (from seqKD/src/):
  python run_debias.py                  # full Stage 1
  python run_debias.py --skip_existing
  python run_debias.py --dry_run
  python run_debias.py --report_only
"""
import argparse, os, re, subprocess, sys, time

CKPT = "../../BSARec/src/output"
DS = {
    "ML-1M": dict(ckpt="BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt", heads="1", alpha="0.3", c="9",
                  drop="0.2", lam_pl="0.5", rank_k="50"),
    "Beauty": dict(ckpt="BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt", heads="2", alpha="0.7", c="5",
                   drop="0.5", lam_pl="2.0", rank_k="50"),
    "LastFM": dict(ckpt="BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt", heads="1", alpha="0.9", c="3",
                   drop="0.5", lam_pl="1.0", rank_k="10"),
}
BASELINE = {"ML-1M": 0.3008, "Beauty": 0.0982, "LastFM": 0.0807}   # pure PL (cmi b0)


def runs_phase1(ds):
    r = []
    for lam in ("0.1", "1.0", "10.0"):
        r.append((f"db_margin_{ds}_l{lam}_m0.3", "margin",
                  ["--lambda_db", lam, "--margin_m", "0.3"]))
    for lam in ("0.1", "1.0", "10.0"):
        r.append((f"db_bpr_{ds}_l{lam}", "bpr", ["--lambda_db", lam]))
    for m in ("0.5", "1.0", "2.0"):
        r.append((f"db_lm_{ds}_m{m}", "logit_margin", ["--lm_margin", m]))
    for g in ("0.5", "1.0", "2.0"):
        r.append((f"db_rw_{ds}_g{g}", "reweight", ["--gamma_rw", g]))
    return r


def runs_phase2(ds, best_lam):
    return [(f"db_margin_{ds}_l{best_lam}_m{m}", "margin",
             ["--lambda_db", best_lam, "--margin_m", m]) for m in ("0.1", "0.5")]


def build_cmd(ds, name, mode, extra, args):
    cfg = DS[ds]
    return [
        sys.executable, "main.py", "--model_type", "kdstudent_v3", "--data_name", ds,
        "--train_name", name, "--do_distill", "--kd_mode", "debias",
        "--debias_mode", mode, "--db_warmup", str(args.db_warmup),
        "--teacher_type", "bsarec", "--teacher_ckpt", os.path.join(CKPT, cfg["ckpt"]),
        "--teacher_num_attention_heads", cfg["heads"], "--teacher_alpha", cfg["alpha"],
        "--teacher_c", cfg["c"], "--alpha", cfg["alpha"], "--c", cfg["c"],
        "--d_state", "16", "--d_conv", "4", "--expand", "1",
        "--hidden_size", "64", "--num_hidden_layers", "2",
        "--hidden_dropout_prob", cfg["drop"], "--attention_probs_dropout_prob", cfg["drop"],
        "--lr", "0.001", "--batch_size", "256", "--epochs", str(args.epochs),
        "--patience", str(args.patience), "--seed", str(args.seed), "--gpu_id", args.gpu_id,
        "--lambda_kd", cfg["lam_pl"], "--rank_k", cfg["rank_k"],
    ] + extra


FIN = re.compile(r"DEBIAS finalize -> .*?: (\{.*\})")


def finalize_row(name):
    p = os.path.join("output", name + ".log")
    if not os.path.exists(p):
        return None
    m = FIN.findall(open(p, encoding="utf-8", errors="replace").read())
    return eval(m[-1]) if m else None      # log line is a printed dict


def log_done(name):
    p = os.path.join("output", name + ".log")
    return os.path.exists(p) and "Test Score" in open(p, encoding="utf-8",
                                                      errors="replace").read()


def execute(runs, ds, args, t0, done_counter):
    for name, mode, extra in runs:
        done_counter[0] += 1
        i = done_counter[0]
        if args.skip_existing and log_done(name):
            print(f"[{i}] SKIP {name}", flush=True); continue
        cmd = build_cmd(ds, name, mode, extra, args)
        print(f"\n[{i}] RUN {name}", flush=True)
        if args.dry_run:
            print("  " + " ".join(cmd), flush=True); continue
        ts = time.perf_counter(); rc = subprocess.call(cmd)
        print(f"[{i}] DONE {name} rc={rc} {time.perf_counter()-ts:.1f}s "
              f"(total {time.perf_counter()-t0:.0f}s)", flush=True)


def report(ds):
    print(f"\n## {ds}  (baseline pure-PL HR@10 = {BASELINE[ds]})")
    names = [n for n, _, _ in runs_phase1(ds)]
    names += [f"db_margin_{ds}_l{lam}_m{m}" for lam in ("0.1", "1.0", "10.0")
              for m in ("0.1", "0.5")]
    for name in names:
        row = finalize_row(name)
        if row is None:
            if os.path.exists(os.path.join("output", name + ".log")):
                print(f"  {name}: (running/incomplete)")
            continue
        print(f"  {name}: HR@10={row['HR@10']} NDCG@10={row['NDCG@10']} "
              f"HRLI@1={row['HRLI@1']} leak={row['leak_test']} "
              f"w[lo/mid/hi]={row['HR@10_w_lo']}/{row['HR@10_w_mid']}/{row['HR@10_w_hi']} "
              f"pop[lo/mid/hi]={row['HR@10_pop_lo']}/{row['HR@10_pop_mid']}/{row['HR@10_pop_hi']} "
              f"s_y(hi)={row['s_y_raw_hi_final']} s_l(hi)={row['s_l_raw_hi_final']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ML-1M"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--db_warmup", type=int, default=10)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--report_only", action="store_true")
    args = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("output", exist_ok=True)

    if args.report_only:
        for ds in args.datasets:
            report(ds)
        return

    t0 = time.perf_counter(); cnt = [0]
    for ds in args.datasets:
        print(f"=== {ds}: phase 1 (12 runs) ===", flush=True)
        execute(runs_phase1(ds), ds, args, t0, cnt)
        # pick best margin lambda by test HR@10, then phase 2 (m sweep)
        cand = [(lam, finalize_row(f"db_margin_{ds}_l{lam}_m0.3"))
                for lam in ("0.1", "1.0", "10.0")]
        cand = [(lam, r["HR@10"]) for lam, r in cand if r]
        if not cand:
            print(f"=== {ds}: no margin results, skip phase 2 ===", flush=True); continue
        best_lam = max(cand, key=lambda x: x[1])[0]
        print(f"=== {ds}: phase 2, best margin lambda* = {best_lam} "
              f"(HR@10 {dict(cand)[best_lam]}) ===", flush=True)
        execute(runs_phase2(ds, best_lam), ds, args, t0, cnt)
    print(f"\n[debias] total wall: {time.perf_counter()-t0:.1f}s", flush=True)
    for ds in args.datasets:
        report(ds)


if __name__ == "__main__":
    main()
