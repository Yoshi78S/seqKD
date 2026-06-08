"""Aggregate Tier-1 ablation results into a single markdown.

- Front matter: detailed description of each ablation unit
- Per-dataset: one architecture table + one distillation table

Output: ../ABLATION_RESULTS.md
"""
import glob
import os
import re

DATASETS = ["Beauty", "LastFM", "ML-1M"]
METRICS = ["HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20"]

ARCH_ABLATIONS = [
    "no_pos_emb",
    "no_input_ln",
    "no_ffn",
    "no_block_ln",
    "flat_gru",
]

DIST_STAGES = [
    "d1_standalone",
    "d2_pred_only",
    "d3_hs_only",
    "d4_full",
]

# Pretty labels (for the result tables only; explanations go in the front matter)
ARCH_LABEL = {
    "no_pos_emb":  "no_pos_emb",
    "no_input_ln": "no_input_ln",
    "no_ffn":      "no_ffn",
    "no_block_ln": "no_block_ln",
    "flat_gru":    "flat_gru (HS-KD disabled)",
}
DIST_LABEL = {
    "d1_standalone": "D1: standalone (L_rec only)",
    "d2_pred_only":  "D2: + Pred-KD",
    "d3_hs_only":    "D3: + HS-KD only",
    "d4_full":       "D4: Pred-KD + HS-KD (full)",
}

SCORE_RE = re.compile(
    r"'Epoch':\s*\d+,\s*"
    r"'HR@5':\s*'([\d.]+)',\s*'NDCG@5':\s*'([\d.]+)',\s*"
    r"'HR@10':\s*'([\d.]+)',\s*'NDCG@10':\s*'([\d.]+)',\s*"
    r"'HR@20':\s*'([\d.]+)',\s*'NDCG@20':\s*'([\d.]+)'"
)


def parse_log(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
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


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Pull scores
    arch_scores = {}  # (ds, variant) -> dict
    dist_scores = {}  # (ds, stage) -> dict
    for ds in DATASETS:
        for v in ARCH_ABLATIONS:
            arch_scores[(ds, v)] = parse_log(f"output/kdstudent_{ds}_abl_arch_{v}.log")
        for s in DIST_STAGES:
            dist_scores[(ds, s)] = parse_log(f"output/kdstudent_{ds}_abl_dist_{s}.log")

    lines = []
    # ----- Front matter -----
    lines.append("# KDStudent Tier-1 Ablation Results\n")
    lines.append("27 runs: 5 architecture ablations + 4 distillation stages, "
                 "each repeated across 3 datasets. All non-ablated "
                 "hyperparameters are fixed to the KDStudent grid winner per "
                 "dataset (Beauty: λ_pred=2.0, T=1.0, λ_hs=0.05, layer=last; "
                 "LastFM: λ_pred=2.0, T=2.0, λ_hs=0.2, layer=last; "
                 "ML-1M: λ_pred=0.5, T=5.0, λ_hs=0.05, layer=last).\n")

    lines.append("## Ablation Units\n")
    lines.append("### Architecture (5 variants, one component removed each)\n")
    lines.append("| # | Variant | What is removed | Distillation kept |")
    lines.append("|---|---|---|---|")
    lines.append("| A1 | `no_pos_emb`  | `item_emb + pos_emb` の位置埋め込み加算をスキップ（item_emb のみ）| 全て（Pred + HS）|")
    lines.append("| A2 | `no_input_ln` | 入力後の `LayerNorm + Dropout` をスキップ | 全て（Pred + HS）|")
    lines.append("| A3 | `no_ffn`      | 各 StudentBlock 内の FFN（GELU + Linear×2 + Dropout + 残差 + LN）を恒等写像化 | 全て（Pred + HS）|")
    lines.append("| A4 | `no_block_ln` | 各 StudentBlock 内の GRU 後 LayerNorm を除去 | 全て（Pred + HS）|")
    lines.append("| A5 | `flat_gru`    | StudentBlock × N を `nn.GRU(num_layers=N)` 1 つに置換（block 構造廃止、FFN/LN も自動消失）| **Pred-KD のみ**（per-block hook 不可で HS-KD 不能）|")
    lines.append("")
    lines.append("### Distillation (4 stages, loss term の段階的追加)\n")
    lines.append("| # | Stage | Loss | Triggered trainer |")
    lines.append("|---|---|---|---|")
    lines.append("| D1 | `d1_standalone` | `L_rec` | `Trainer` (no teacher) |")
    lines.append("| D2 | `d2_pred_only`  | `L_rec + λ_pred · L_pred` | `DistillTrainer` |")
    lines.append("| D3 | `d3_hs_only`    | `L_rec + λ_hs · L_hs`（`λ_pred = 0`）| `KDStudentDistillTrainer` |")
    lines.append("| D4 | `d4_full`       | `L_rec + λ_pred · L_pred + λ_hs · L_hs` | `KDStudentDistillTrainer` |")
    lines.append("")
    lines.append("D4 (full) は grid 探索のベスト設定と同一構成（同じ seed=42 で再学習した参照）。")
    lines.append("アーキ ablation のリファレンスはこの D4 と読み替える。\n")

    # ----- Per-dataset results -----
    for ds in DATASETS:
        lines.append(f"## {ds}\n")

        # Architecture table
        lines.append(f"### Architecture ablation\n")
        lines.append("| Variant | " + " | ".join(METRICS) + " |")
        lines.append("|" + "---|" * (len(METRICS) + 1))
        # Show D4 as the reference at the top
        d4 = dist_scores.get((ds, "d4_full"))
        if d4 is not None:
            cells = ["**ref (D4 full)**"] + [f"{d4[m]:.4f}" for m in METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        for v in ARCH_ABLATIONS:
            sc = arch_scores.get((ds, v))
            label = ARCH_LABEL[v]
            if sc is None:
                cells = [label] + ["—"] * len(METRICS)
            else:
                cells = [label] + [f"{sc[m]:.4f}" for m in METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        # Distillation table
        lines.append(f"### Distillation ablation\n")
        lines.append("| Stage | " + " | ".join(METRICS) + " |")
        lines.append("|" + "---|" * (len(METRICS) + 1))
        for s in DIST_STAGES:
            sc = dist_scores.get((ds, s))
            label = DIST_LABEL[s]
            if sc is None:
                cells = [label] + ["—"] * len(METRICS)
            else:
                cells = [label] + [f"{sc[m]:.4f}" for m in METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    out_path = "../ABLATION_RESULTS.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")
    miss = [k for k, v in {**arch_scores, **dist_scores}.items() if v is None]
    if miss:
        print(f"  missing: {miss}")
    else:
        print(f"  all {len(arch_scores) + len(dist_scores)} runs parsed")


if __name__ == "__main__":
    main()
