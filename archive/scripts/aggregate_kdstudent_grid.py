"""Summarize KDStudent grid-search results vs baselines.

For each dataset, produce one comparison table containing:
  * 12 baseline models (best HP from BSARec grid logs)
  * the KDStudent best HP combination
Sorted ascending by HR@10 (best at bottom). Column-wise maximum bolded.

Also emits a separate KDStudent best-HP summary.

Output: ../KD_RESULTS.md
"""
import glob
import os
import re

DATASETS = ["Beauty", "LastFM", "ML-1M"]
METRICS = ["HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20"]
BASELINE_MODELS = [
    "BSARec", "DuoRec", "SIGMA",
    "SASRec", "BERT4Rec", "FEARec", "FMLPRec", "GRU4Rec",
    "Mamba4Rec", "LRURec", "ICSRec", "ICLRec",
]
BSAREC_OUT = "../../BSARec/src/output"
SEQKD_OUT = "output"

SCORE_RE = re.compile(
    r"'Epoch':\s*\d+,\s*"
    r"'HR@5':\s*'([\d.]+)',\s*'NDCG@5':\s*'([\d.]+)',\s*"
    r"'HR@10':\s*'([\d.]+)',\s*'NDCG@10':\s*'([\d.]+)',\s*"
    r"'HR@20':\s*'([\d.]+)',\s*'NDCG@20':\s*'([\d.]+)'"
)
TIMING_TEST_RE = re.compile(r"TIMING\s+test\s+([\d.]+)")
KD_NAME_RE = re.compile(
    r"kdstudent_[^_]+(?:-[^_]+)?_grid_lp([\d.]+)_t([\d.]+)_lhs([\d.]+)(?:_(last|all))?$"
)


def parse_log(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    idx = text.rfind("Test Score")
    if idx < 0:
        return None, None
    m = SCORE_RE.findall(text[idx:])
    if not m:
        return None, None
    hr5, ndcg5, hr10, ndcg10, hr20, ndcg20 = m[-1]
    score = {
        "HR@5": float(hr5), "NDCG@5": float(ndcg5),
        "HR@10": float(hr10), "NDCG@10": float(ndcg10),
        "HR@20": float(hr20), "NDCG@20": float(ndcg20),
    }
    timing = TIMING_TEST_RE.findall(text[idx:]) or TIMING_TEST_RE.findall(text)
    ttime = float(timing[-1]) if timing else None
    return score, ttime


def sort_key(score, stem):
    return (
        -score["HR@10"], -score["NDCG@10"], -score["HR@5"],
        -score["NDCG@5"], -score["HR@20"], -score["NDCG@20"], stem,
    )


def best_baseline(model, dataset):
    runs = []
    for path in sorted(glob.glob(f"{BSAREC_OUT}/{model}_{dataset}_grid_*.log")):
        score, ttime = parse_log(path)
        if score is None:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        runs.append((score, ttime, stem))
    if not runs:
        return None
    runs.sort(key=lambda r: sort_key(r[0], r[2]))
    return runs[0]


def best_kdstudent(dataset):
    runs = []
    for path in sorted(glob.glob(f"{SEQKD_OUT}/kdstudent_{dataset}_grid_*.log")):
        score, ttime = parse_log(path)
        if score is None:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        runs.append((score, ttime, stem))
    if not runs:
        return None
    runs.sort(key=lambda r: sort_key(r[0], r[2]))
    return runs[0]


def parse_kd_hp(stem):
    """kdstudent_Beauty_grid_lp2.0_t1.0_lhs0.05_last → (2.0, 1.0, 0.05, 'last')."""
    m = KD_NAME_RE.match(stem)
    if not m:
        return None
    lp, T, lhs = float(m.group(1)), float(m.group(2)), float(m.group(3))
    layer = m.group(4) or "—"
    return lp, T, lhs, layer


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Gather data per dataset: list of (label, score, test_time)
    per_ds = {}
    kd_bests = {}
    for ds in DATASETS:
        rows = []
        for m in BASELINE_MODELS:
            res = best_baseline(m, ds)
            if res is not None:
                score, ttime, _ = res
                rows.append((m, score, ttime))
        kd_res = best_kdstudent(ds)
        if kd_res is not None:
            score, ttime, stem = kd_res
            rows.append(("KDStudent", score, ttime))
            kd_bests[ds] = (score, ttime, stem)
        per_ds[ds] = rows

    lines = []
    lines.append("# KDStudent vs Baselines\n")
    lines.append("KDStudent grid (λ_pred × T × λ_hs × layer = 84 configs per "
                 "dataset) vs the 12 baseline models. For each dataset the "
                 "table is sorted ascending by HR@10, so the strongest model "
                 "sits at the bottom. The column-wise maximum per metric is "
                 "**bolded**.\n")
    lines.append("- KDStudent logs: `seqKD/src/output/kdstudent_<DS>_grid_*.log`")
    lines.append("- Baseline logs : `BSARec/src/output/<Model>_<DS>_grid_*.log`\n")

    for ds in DATASETS:
        rows = per_ds[ds]
        if not rows:
            lines.append(f"## {ds}\n\n*(no completed runs)*\n")
            continue
        col_max = {m: max(score[m] for _, score, _ in rows) for m in METRICS}
        rows.sort(key=lambda r: r[1]["HR@10"])

        lines.append(f"## {ds}\n")
        lines.append("| Model | " + " | ".join(METRICS) + " | Inf. time (s) |")
        lines.append("|" + "---|" * (len(METRICS) + 2))
        for label, score, ttime in rows:
            cells = [f"**{label}**" if label == "KDStudent" else label]
            for m in METRICS:
                v = score[m]
                cells.append(f"**{v:.4f}**" if v == col_max[m] else f"{v:.4f}")
            cells.append(f"{ttime:.4f}" if ttime is not None else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # KDStudent best HP summary
    lines.append("## KDStudent Best Hyperparameters\n")
    lines.append("| Dataset | λ_pred | T | λ_hs | layer | HR@10 | NDCG@10 |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        if ds not in kd_bests:
            lines.append(f"| {ds} | — | — | — | — | — | — |")
            continue
        score, _, stem = kd_bests[ds]
        hp = parse_kd_hp(stem)
        if hp is None:
            lines.append(f"| {ds} | (parse failed) | | | | "
                         f"{score['HR@10']:.4f} | {score['NDCG@10']:.4f} |")
            continue
        lp, T, lhs, layer = hp
        lines.append(f"| {ds} | {lp} | {T} | {lhs} | {layer} "
                     f"| {score['HR@10']:.4f} | {score['NDCG@10']:.4f} |")
    lines.append("")

    out_path = "../KD_RESULTS.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")
    print(f"  Baselines covered: "
          f"{sum(len(r) - (1 if ds in kd_bests else 0) for ds, r in per_ds.items())}")
    print(f"  KDStudent bests:   {len(kd_bests)} / {len(DATASETS)}")


if __name__ == "__main__":
    main()
