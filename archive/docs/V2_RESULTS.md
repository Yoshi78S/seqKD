# KDStudent v2 Initial Results

GLINT-RU-inspired v2 architecture (CausalConv1D + SelectiveGate + GatedMLP)
evaluated against v1 at the same per-dataset hyperparameters (the v1 grid
winners). Goal: measure the pure architecture swap effect, before any v2-
specific HP re-search.

- v2 logs    : `seqKD/src/output/kdstudent_v2_<DS>_initial.log`
- v1 winners : `seqKD/src/output/kdstudent_<DS>_grid_lp*_t*_lhs*_last.log`
- Settings   : λ_pred, T, λ_hs, layer = per-DS v1 grid winner; same seed=42.

## Hyperparameters used (v1 winners, applied to v2)

| Dataset | λ_pred | T | λ_hs | layer | Dropout |
|---|---|---|---|---|---|
| Beauty | 2.0 | 1.0 | 0.05 | last | 0.5 |
| LastFM | 2.0 | 2.0 | 0.2  | last | 0.5 |
| ML-1M  | 0.5 | 5.0 | 0.05 | last | 0.2 |

## Performance — v2 vs v1 grid winner

### Beauty

| Metric | v1 best | v2 initial | Δ (absolute) | Δ (%) |
|---|---|---|---|---|
| HR@5    | 0.0537 | 0.0496 | -0.0041 | -7.6% |
| NDCG@5  | 0.0366 | 0.0333 | -0.0033 | -9.0% |
| HR@10   | **0.0820** | **0.0751** | -0.0069 | **-8.4%** |
| NDCG@10 | 0.0458 | 0.0415 | -0.0043 | -9.4% |
| HR@20   | 0.1167 | 0.1109 | -0.0058 | -5.0% |
| NDCG@20 | 0.0545 | 0.0505 | -0.0040 | -7.3% |

### LastFM

| Metric | v1 best | v2 initial | Δ (absolute) | Δ (%) |
|---|---|---|---|---|
| HR@5    | 0.0394 | 0.0422 | +0.0028 | **+7.1%** |
| NDCG@5  | 0.0254 | 0.0271 | +0.0017 | +6.7% |
| HR@10   | **0.0670** | **0.0624** | -0.0046 | **-6.9%** |
| NDCG@10 | 0.0340 | 0.0336 | -0.0004 | -1.2% |
| HR@20   | 0.0945 | 0.0945 | 0.0000  | 0.0% |
| NDCG@20 | 0.0409 | 0.0417 | +0.0008 | +2.0% |

### ML-1M

| Metric | v1 best | v2 initial | Δ (absolute) | Δ (%) |
|---|---|---|---|---|
| HR@5    | 0.2104 | 0.1965 | -0.0139 | -6.6% |
| NDCG@5  | 0.1431 | 0.1317 | -0.0114 | -8.0% |
| HR@10   | **0.3026** | **0.2882** | -0.0144 | **-4.8%** |
| NDCG@10 | 0.1728 | 0.1613 | -0.0115 | -6.7% |
| HR@20   | 0.4152 | 0.3965 | -0.0187 | -4.5% |
| NDCG@20 | 0.2012 | 0.1886 | -0.0126 | -6.3% |

## Cost (training + inference)

| Dataset | Model | n_params | epochs run | best epoch | train sec/epoch | test sec | wall train (s) |
|---|---|---|---|---|---|---|---|
| Beauty | v1 | 893,696 | 48  | 37 | 5.5539 | 2.6881 | — |
| Beauty | v2 | 910,464 (+1.9%) | 61  | 50 | 5.5407 | 2.7122 (+0.9%) | 503.3 |
| LastFM | v1 | 352,576 | 48  | 37 | 0.9946 | 0.1444 | — |
| LastFM | v2 | 369,344 (+4.8%) | 46  | 35 | 1.0365 (+4.2%) | 0.1469 (+1.7%) | 54.7 |
| ML-1M  | v1 | 337,856 | 104 | 93 | 9.7121 | 0.4172 | — |
| ML-1M  | v2 | 354,624 (+5.0%) | 47  | 36 | 9.7935 (+0.8%) | 0.4257 (+2.0%) | 480.5 |

## Comparison against the 12-model baseline (HR@10)

Reading off [`BSARec/BASELINE_RESULTS.md`](../BSARec/BASELINE_RESULTS.md):

### Beauty (v2 = 0.0751)

| Place | Model | HR@10 |
|---|---|---|
| ↑ | BSARec (teacher) | 0.0985 |
| ↑ | FEARec | 0.0986 |
| ↑ | BERT4Rec | 0.0719 |
| **v2 falls here** | **KDStudent v2** | **0.0751** |
| ↓ | FMLPRec | 0.0625 |
| ↓ | SASRec | 0.0563 |

v2 ranks 5th of 13 on Beauty; below v1 best (0.0820 = 4th) but still above
BERT4Rec / FMLPRec / SASRec / GRU4Rec.

### LastFM (v2 = 0.0624)

| Place | Model | HR@10 |
|---|---|---|
| ↑ | BSARec (teacher) | 0.0761 |
| ↑ | ICSRec | 0.0688 |
| ↑ | SASRec | 0.0679 |
| ↑ | DuoRec / LRURec / KDStudent v1 | 0.0670 |
| ↑ | FEARec / BERT4Rec / ICLRec | 0.0642 |
| **v2 falls here** | **KDStudent v2** | **0.0624** |
| ↓ | SIGMA | 0.0615 |
| ↓ | Mamba4Rec | 0.0587 |

v2 ranks 11th of 13 on LastFM. Worse than most baselines.

### ML-1M (v2 = 0.2882)

| Place | Model | HR@10 |
|---|---|---|
| ↑ | SIGMA | 0.3084 |
| ↑ | DuoRec | 0.3076 |
| ↑ | FEARec / KDStudent v1 | 0.3058 / 0.3026 |
| ↑ | Mamba4Rec | 0.3005 |
| ↑ | ICSRec | 0.2983 |
| ↑ | LRURec | 0.2901 |
| ↑ | ICLRec | 0.2816 |
| **v2 falls here** | **KDStudent v2** | **0.2882** |
| ↓ | BSARec (teacher) | 0.2800 |
| ↓ | BERT4Rec | 0.2331 |

v2 ranks 8th of 13 on ML-1M, **still above the BSARec teacher (0.2800)**,
but below v1 best.

## Summary table

| Dataset | v1 HR@10 | v2 HR@10 | Δ | v2 rank / 13 | v1 rank / 13 |
|---|---|---|---|---|---|
| Beauty | 0.0820 | 0.0751 | -8.4% | 5 | 5 |
| LastFM | 0.0670 | 0.0624 | -6.9% | 11 | 10 |
| ML-1M  | 0.3026 | 0.2882 | -4.8% | 8 | 4 |

## Parameter / runtime overhead vs v1

| Dataset | params Δ | train/epoch Δ | inference Δ |
|---|---|---|---|
| Beauty | +16,768 (+1.9%) | -0.0132s (-0.2%) | +0.0241s (+0.9%) |
| LastFM | +16,768 (+4.8%) | +0.0419s (+4.2%) | +0.0025s (+1.7%) |
| ML-1M  | +16,768 (+5.0%) | +0.0814s (+0.8%) | +0.0085s (+2.0%) |

Per-block overhead is constant +8,384 params × 2 blocks = +16,768 across all
datasets; the % varies only because item_embedding sizes differ. Runtime
overhead is ≤ 5% throughout.

## Convergence

| Dataset | v1 epochs_run / best | v2 epochs_run / best |
|---|---|---|
| Beauty | 48 / 37 | 61 / 50 |
| LastFM | 48 / 37 | 46 / 35 |
| ML-1M  | 104 / 93 | 47 / 36 |

ML-1M converges much earlier with v2 (47 vs 104 epochs) — possibly because
the heavier per-block computation produces a more expressive representation
that fits the training set faster, at the cost of generalisation.
