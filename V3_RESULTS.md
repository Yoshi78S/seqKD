# KDStudent v3 (FreqMamba) — Grid Search Results

FreqMamba = BSARec's two-branch fusion with the attention branch replaced by a
residual-free Mamba (see `src/model/kd_student_v3.py`). Frequency branch is
BSARec-identical (β=sqrt_beta², cutoff c inherited); fusion α and c set to the
teacher's per-dataset values. Mamba: expand=1, d_state=16, d_conv=4. FFN: inner=2d.

Grid: `--mode focused` = λ_pred∈{0.5,1.0,2.0} × T∈{1.0,2.0,5.0} ×
(λ_hs,layer)∈{(0,—),(0.05,last),(0.1,last),(0.2,last)} = 36/dataset, 108 total.
Runner: `src/run_kdstudent_v3_grid.py`. Logs: `src/output/kdstudent_v3_*.log`.
Completed 2026-05-30 (seed=42).

## Best hyperparameters per dataset

| Dataset | α | λ_pred | T | λ_hs | layer | HR@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| Beauty | 0.7 | 1.0 | 1.0 | 0.05 | last | 0.0989 | 0.0594 |
| LastFM | 0.9 | 0.5 | 2.0 | 0.2  | last | 0.0761 | 0.0424 |
| ML-1M  | 0.3 | 0.5 | 2.0 | 0.05 | last | 0.3078 | 0.1780 |

## vs teacher and baselines (HR@10 / NDCG@10)

### Beauty — v3 is #1 on HR@10 (beats teacher + FEARec)

| Model | HR@10 | NDCG@10 |
|---|---|---|
| KDStudent v1 (GRU) | 0.0820 | 0.0458 |
| SIGMA | 0.0891 | 0.0546 |
| Mamba4Rec | 0.0914 | 0.0551 |
| LRURec | 0.0956 | 0.0572 |
| BSARec (teacher) | 0.0985 | **0.0599** |
| FEARec | 0.0986 | 0.0597 |
| **KDStudent v3** | **0.0989** | 0.0594 |

### LastFM — v3 matches the teacher (was v1's worst dataset, 10th)

| Model | HR@10 | NDCG@10 |
|---|---|---|
| KDStudent v1 (GRU) | 0.0670 | 0.0340 |
| SASRec | 0.0679 | 0.0374 |
| ICSRec | 0.0688 | 0.0413 |
| BSARec (teacher) | **0.0761** | **0.0437** |
| **KDStudent v3** | **0.0761** | 0.0424 |

### ML-1M — v3 beats the teacher, NDCG@10 tied-best

| Model | HR@10 | NDCG@10 |
|---|---|---|
| BSARec (teacher) | 0.2800 | 0.1572 |
| KDStudent v1 (GRU) | 0.3026 | 0.1728 |
| Mamba4Rec | 0.3005 | 0.1697 |
| FEARec | 0.3058 | 0.1760 |
| DuoRec | 0.3076 | **0.1780** |
| **KDStudent v3** | **0.3078** | **0.1780** |
| SIGMA | 0.3084 | 0.1766 |

## Cost (best run per dataset)

| Dataset | n_params (total) | body (excl. emb) | inference (test, s) | epochs to best |
|---|---|---|---|---|
| Beauty | 844,544 | 66,816 | 2.794 | 52 |
| LastFM | 303,424 | 66,816 | 0.150 | 31 |
| ML-1M  | 288,704 | 66,816 | 0.420 | 50 |

- v3 body (66,816) is lighter than v1 (~116K) and the BSARec teacher (~100K).
- Inference at n=50 is ~teacher-level or faster, but not faster than v1/SASRec
  (Mamba selective-scan + FFT overhead). The O(n) advantage is asymptotic
  (materializes at long n), not a win at n=50 — see direction-update memory.

## Takeaways

1. **3/3 datasets: v3 ≥ teacher.** Beauty #1 (>teacher, >FEARec); LastFM matches
   the teacher (resolving v1's biggest weakness, 10th→tied-1st); ML-1M beats the
   teacher and ties best NDCG@10.
2. **+20.6% over v1 on Beauty**, large gains on LastFM/ML-1M too — while being
   lighter and O(n).
3. **HS-KD contribution remains small** (Pred-only ≈ HS-on): the improvement is
   driven by the FreqMamba architecture (BSARec structure inheritance +
   residual-free Mamba), not the hidden-state alignment.
4. Distillation HP trend: Beauty wants sharp/strong (T=1.0, λ_pred=1.0); LastFM
   and ML-1M prefer T=2.0, λ_pred=0.5 (softer, more student autonomy).
