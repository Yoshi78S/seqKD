"""Aggregate KDStudent v2 ablation results.

Front matter explains each v2-specific component being ablated; per-dataset
tables show all 6 metrics with the reference (v2 initial, all components on)
at the top.

Output: ../ABLATION_V2_RESULTS.md
"""
import glob
import os
import re

DATASETS = ["Beauty", "LastFM", "ML-1M"]
METRICS = ["HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20"]

ABLATIONS = [
    "no_pos_emb",
    "no_input_ln",
    "no_block_ln",
    "no_conv",
    "no_gate",
    "no_gated_mlp",
]

ABL_LABEL = {
    "no_pos_emb":  "no_pos_emb",
    "no_input_ln": "no_input_ln",
    "no_block_ln": "no_block_ln",
    "no_conv":     "no_conv",
    "no_gate":     "no_gate (HS-KD hooks GRU)",
    "no_gated_mlp": "no_gated_mlp",
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
    hr5, n5, hr10, n10, hr20, n20 = m[-1]
    return {
        "HR@5": float(hr5), "NDCG@5": float(n5),
        "HR@10": float(hr10), "NDCG@10": float(n10),
        "HR@20": float(hr20), "NDCG@20": float(n20),
    }


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    ref_scores = {}
    abl_scores = {}
    for ds in DATASETS:
        ref_scores[ds] = parse_log(f"output/kdstudent_v2_{ds}_initial.log")
        for v in ABLATIONS:
            abl_scores[(ds, v)] = parse_log(
                f"output/kdstudent_v2_{ds}_abl_arch_{v}.log")

    lines = []
    # ── Front matter ──
    lines.append("# KDStudent v2 Architecture Ablation Results\n")
    lines.append("18 runs: 6 architecture removals × 3 datasets. All non-ablated "
                 "hyperparameters fixed to the v1 grid winner per dataset "
                 "(Beauty: λ_pred=2.0, T=1.0, λ_hs=0.05, layer=last; "
                 "LastFM: λ_pred=2.0, T=2.0, λ_hs=0.2, layer=last; "
                 "ML-1M: λ_pred=0.5, T=5.0, λ_hs=0.05, layer=last). "
                 "Loss: `L_rec + λ_pred · L_pred + λ_hs · L_hs` (no CDD).\n")
    lines.append("- Reference: `kdstudent_v2_<DS>_initial` (full v2 architecture, all components ON)")
    lines.append("- Ablation logs: `kdstudent_v2_<DS>_abl_arch_<variant>`")
    lines.append("- Script: `seqKD/src/run_ablation_v2.py`\n")

    lines.append("## Ablation Units (v2-specific)\n")
    lines.append("| # | Variant | What is removed | HS-KD hook target |")
    lines.append("|---|---|---|---|")
    lines.append("| V1 | `no_pos_emb`    | `item_emb + pos_emb` の位置埋め込み加算をスキップ | SelectiveGate |")
    lines.append("| V2 | `no_input_ln`   | 入力後の `LayerNorm + Dropout` をスキップ        | SelectiveGate |")
    lines.append("| V3 | `no_block_ln`   | 各 StudentBlockV2 の GRU 後 LayerNorm を除去      | SelectiveGate |")
    lines.append("| V4 | `no_conv`       | Linear → CausalConv1D を除去（GRU が input を直接受ける） | SelectiveGate |")
    lines.append("| V5 | `no_gate`       | SelectiveGate を除去（GRU 出力が直接通過）        | **GRU**（フォールバック） |")
    lines.append("| V6 | `no_gated_mlp`  | GatedMLP を除去（ブロックは gate+LN で終端）       | SelectiveGate |")
    lines.append("")

    # ── Per-dataset tables ──
    for ds in DATASETS:
        lines.append(f"## {ds}\n")
        lines.append("| Variant | " + " | ".join(METRICS) + " |")
        lines.append("|" + "---|" * (len(METRICS) + 1))
        ref = ref_scores[ds]
        if ref is not None:
            cells = ["**ref (v2 initial)**"] + [f"{ref[m]:.4f}" for m in METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        for v in ABLATIONS:
            sc = abl_scores.get((ds, v))
            label = ABL_LABEL[v]
            if sc is None:
                cells = [label] + ["—"] * len(METRICS)
            else:
                cells = [label] + [f"{sc[m]:.4f}" for m in METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    out_path = "../ABLATION_V2_RESULTS.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")
    miss = [k for k, v in {**{(ds, "ref"): ref_scores[ds] for ds in DATASETS},
                            **abl_scores}.items() if v is None]
    if miss:
        print(f"  missing: {miss}")
    else:
        print(f"  all {3 + len(abl_scores)} runs parsed (3 refs + 18 ablations)")


if __name__ == "__main__":
    main()
