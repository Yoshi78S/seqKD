# KDStudent v2 Architecture Ablation Results

18 runs: 6 architecture removals × 3 datasets. All non-ablated hyperparameters fixed to the v1 grid winner per dataset (Beauty: λ_pred=2.0, T=1.0, λ_hs=0.05, layer=last; LastFM: λ_pred=2.0, T=2.0, λ_hs=0.2, layer=last; ML-1M: λ_pred=0.5, T=5.0, λ_hs=0.05, layer=last). Loss: `L_rec + λ_pred · L_pred + λ_hs · L_hs` (no CDD).

- Reference: `kdstudent_v2_<DS>_initial` (full v2 architecture, all components ON)
- Ablation logs: `kdstudent_v2_<DS>_abl_arch_<variant>`
- Script: `seqKD/src/run_ablation_v2.py`

## Ablation Units (v2-specific)

| # | Variant | What is removed | HS-KD hook target |
|---|---|---|---|
| V1 | `no_pos_emb`    | `item_emb + pos_emb` の位置埋め込み加算をスキップ | SelectiveGate |
| V2 | `no_input_ln`   | 入力後の `LayerNorm + Dropout` をスキップ        | SelectiveGate |
| V3 | `no_block_ln`   | 各 StudentBlockV2 の GRU 後 LayerNorm を除去      | SelectiveGate |
| V4 | `no_conv`       | Linear → CausalConv1D を除去（GRU が input を直接受ける） | SelectiveGate |
| V5 | `no_gate`       | SelectiveGate を除去（GRU 出力が直接通過）        | **GRU**（フォールバック） |
| V6 | `no_gated_mlp`  | GatedMLP を除去（ブロックは gate+LN で終端）       | SelectiveGate |

## Beauty

| Variant | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| **ref (v2 initial)** | 0.0496 | 0.0333 | 0.0751 | 0.0415 | 0.1109 | 0.0505 |
| no_pos_emb | 0.0495 | 0.0323 | 0.0733 | 0.0399 | 0.1081 | 0.0487 |
| no_input_ln | 0.0505 | 0.0340 | 0.0766 | 0.0424 | 0.1113 | 0.0511 |
| no_block_ln | 0.0523 | 0.0351 | 0.0753 | 0.0425 | 0.1109 | 0.0514 |
| no_conv | 0.0554 | 0.0369 | 0.0785 | 0.0443 | 0.1135 | 0.0531 |
| no_gate (HS-KD hooks GRU) | 0.0510 | 0.0345 | 0.0759 | 0.0426 | 0.1100 | 0.0512 |
| no_gated_mlp | 0.0502 | 0.0339 | 0.0753 | 0.0419 | 0.1089 | 0.0504 |

## LastFM

| Variant | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| **ref (v2 initial)** | 0.0422 | 0.0271 | 0.0624 | 0.0336 | 0.0945 | 0.0417 |
| no_pos_emb | 0.0367 | 0.0250 | 0.0615 | 0.0330 | 0.0963 | 0.0416 |
| no_input_ln | 0.0385 | 0.0248 | 0.0523 | 0.0294 | 0.0844 | 0.0374 |
| no_block_ln | 0.0413 | 0.0251 | 0.0560 | 0.0298 | 0.0927 | 0.0391 |
| no_conv | 0.0404 | 0.0276 | 0.0560 | 0.0326 | 0.0972 | 0.0429 |
| no_gate (HS-KD hooks GRU) | 0.0349 | 0.0238 | 0.0596 | 0.0315 | 0.0890 | 0.0388 |
| no_gated_mlp | 0.0394 | 0.0267 | 0.0560 | 0.0320 | 0.0991 | 0.0429 |

## ML-1M

| Variant | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| **ref (v2 initial)** | 0.1965 | 0.1317 | 0.2882 | 0.1613 | 0.3965 | 0.1886 |
| no_pos_emb | 0.2007 | 0.1338 | 0.2889 | 0.1624 | 0.3990 | 0.1902 |
| no_input_ln | 0.1894 | 0.1286 | 0.2818 | 0.1584 | 0.3879 | 0.1853 |
| no_block_ln | 0.2041 | 0.1381 | 0.2921 | 0.1663 | 0.3942 | 0.1921 |
| no_conv | 0.2048 | 0.1395 | 0.2969 | 0.1690 | 0.4076 | 0.1970 |
| no_gate (HS-KD hooks GRU) | 0.1927 | 0.1311 | 0.2969 | 0.1646 | 0.4035 | 0.1914 |
| no_gated_mlp | 0.1970 | 0.1327 | 0.2871 | 0.1617 | 0.4036 | 0.1910 |
