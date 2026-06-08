# KDStudent Tier-1 Ablation Results

27 runs: 5 architecture ablations + 4 distillation stages, each repeated across 3 datasets. All non-ablated hyperparameters are fixed to the KDStudent grid winner per dataset (Beauty: λ_pred=2.0, T=1.0, λ_hs=0.05, layer=last; LastFM: λ_pred=2.0, T=2.0, λ_hs=0.2, layer=last; ML-1M: λ_pred=0.5, T=5.0, λ_hs=0.05, layer=last).

## Ablation Units

### Architecture (5 variants, one component removed each)

| # | Variant | What is removed | Distillation kept |
|---|---|---|---|
| A1 | `no_pos_emb`  | `item_emb + pos_emb` の位置埋め込み加算をスキップ（item_emb のみ）| 全て（Pred + HS）|
| A2 | `no_input_ln` | 入力後の `LayerNorm + Dropout` をスキップ | 全て（Pred + HS）|
| A3 | `no_ffn`      | 各 StudentBlock 内の FFN（GELU + Linear×2 + Dropout + 残差 + LN）を恒等写像化 | 全て（Pred + HS）|
| A4 | `no_block_ln` | 各 StudentBlock 内の GRU 後 LayerNorm を除去 | 全て（Pred + HS）|
| A5 | `flat_gru`    | StudentBlock × N を `nn.GRU(num_layers=N)` 1 つに置換（block 構造廃止、FFN/LN も自動消失）| **Pred-KD のみ**（per-block hook 不可で HS-KD 不能）|

### Distillation (4 stages, loss term の段階的追加)

| # | Stage | Loss | Triggered trainer |
|---|---|---|---|
| D1 | `d1_standalone` | `L_rec` | `Trainer` (no teacher) |
| D2 | `d2_pred_only`  | `L_rec + λ_pred · L_pred` | `DistillTrainer` |
| D3 | `d3_hs_only`    | `L_rec + λ_hs · L_hs`（`λ_pred = 0`）| `KDStudentDistillTrainer` |
| D4 | `d4_full`       | `L_rec + λ_pred · L_pred + λ_hs · L_hs` | `KDStudentDistillTrainer` |

D4 (full) は grid 探索のベスト設定と同一構成（同じ seed=42 で再学習した参照）。
アーキ ablation のリファレンスはこの D4 と読み替える。

## Beauty

### Architecture ablation

| Variant | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| **ref (D4 full)** | 0.0537 | 0.0366 | 0.0820 | 0.0458 | 0.1167 | 0.0545 |
| no_pos_emb | 0.0520 | 0.0349 | 0.0796 | 0.0438 | 0.1150 | 0.0527 |
| no_input_ln | 0.0632 | 0.0447 | 0.0915 | 0.0537 | 0.1255 | 0.0623 |
| no_ffn | 0.0585 | 0.0402 | 0.0855 | 0.0488 | 0.1194 | 0.0573 |
| no_block_ln | 0.0590 | 0.0409 | 0.0863 | 0.0497 | 0.1221 | 0.0587 |
| flat_gru (HS-KD disabled) | 0.0631 | 0.0452 | 0.0902 | 0.0539 | 0.1244 | 0.0626 |

### Distillation ablation

| Stage | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| D1: standalone (L_rec only) | 0.0409 | 0.0267 | 0.0636 | 0.0340 | 0.0931 | 0.0414 |
| D2: + Pred-KD | 0.0544 | 0.0374 | 0.0813 | 0.0461 | 0.1127 | 0.0539 |
| D3: + HS-KD only | 0.0423 | 0.0279 | 0.0674 | 0.0360 | 0.0993 | 0.0440 |
| D4: Pred-KD + HS-KD (full) | 0.0537 | 0.0366 | 0.0820 | 0.0458 | 0.1167 | 0.0545 |

## LastFM

### Architecture ablation

| Variant | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| **ref (D4 full)** | 0.0394 | 0.0254 | 0.0670 | 0.0340 | 0.0945 | 0.0409 |
| no_pos_emb | 0.0367 | 0.0257 | 0.0606 | 0.0334 | 0.0917 | 0.0413 |
| no_input_ln | 0.0431 | 0.0284 | 0.0642 | 0.0351 | 0.0972 | 0.0433 |
| no_ffn | 0.0385 | 0.0247 | 0.0550 | 0.0300 | 0.0927 | 0.0395 |
| no_block_ln | 0.0349 | 0.0226 | 0.0661 | 0.0328 | 0.0945 | 0.0400 |
| flat_gru (HS-KD disabled) | 0.0440 | 0.0312 | 0.0688 | 0.0393 | 0.1028 | 0.0478 |

### Distillation ablation

| Stage | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| D1: standalone (L_rec only) | 0.0266 | 0.0196 | 0.0495 | 0.0271 | 0.0697 | 0.0322 |
| D2: + Pred-KD | 0.0394 | 0.0265 | 0.0624 | 0.0338 | 0.0963 | 0.0423 |
| D3: + HS-KD only | 0.0294 | 0.0195 | 0.0560 | 0.0280 | 0.0917 | 0.0371 |
| D4: Pred-KD + HS-KD (full) | 0.0394 | 0.0254 | 0.0670 | 0.0340 | 0.0945 | 0.0409 |

## ML-1M

### Architecture ablation

| Variant | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| **ref (D4 full)** | 0.2104 | 0.1431 | 0.3026 | 0.1728 | 0.4152 | 0.2012 |
| no_pos_emb | 0.1988 | 0.1324 | 0.2838 | 0.1598 | 0.3967 | 0.1884 |
| no_input_ln | 0.2098 | 0.1425 | 0.2983 | 0.1712 | 0.3993 | 0.1967 |
| no_ffn | 0.2053 | 0.1398 | 0.2881 | 0.1664 | 0.4000 | 0.1946 |
| no_block_ln | 0.2103 | 0.1427 | 0.3010 | 0.1718 | 0.4070 | 0.1985 |
| flat_gru (HS-KD disabled) | 0.1975 | 0.1353 | 0.2924 | 0.1657 | 0.3960 | 0.1918 |

### Distillation ablation

| Stage | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| D1: standalone (L_rec only) | 0.1998 | 0.1327 | 0.2877 | 0.1611 | 0.3990 | 0.1890 |
| D2: + Pred-KD | 0.2071 | 0.1401 | 0.2911 | 0.1671 | 0.4086 | 0.1968 |
| D3: + HS-KD only | 0.1929 | 0.1311 | 0.2874 | 0.1615 | 0.4025 | 0.1905 |
| D4: Pred-KD + HS-KD (full) | 0.2104 | 0.1431 | 0.3026 | 0.1728 | 0.4152 | 0.2012 |
