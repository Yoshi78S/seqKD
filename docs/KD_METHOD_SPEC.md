# Current Distillation Method — FreqMamba v3 (authoritative spec)

Verified against code (trainers.py, kd_student_v3.py, utils.py, run_kdstudent_v3_grid.py,
main.py, _modules.py, _abstract_model.py). Student = FreqMamba v3
(`model/kd_student_v3.py`, `kdstudent_v3`). Teacher = BSARec (frozen).

## Total objective

```
L = L_rec  +  λ_pred · L_pred  +  λ_hs · L_hs   ( + λ_cdd · L_cdd )
```
CDD term is dead code in all v3 runs (λ_cdd ≡ 0). Trainer routing (main.py:50,63-69):
`--do_hs_distill` + kdstudent_v3 → `KDStudentDistillTrainer` (full L). Only `--do_distill`
→ `DistillTrainer` (L_rec + λ_pred·L_pred). The λ_hs=0 grid cells take the latter path.

## L_rec — recommendation loss  (trainers.py:505)
Full-vocab cross-entropy on the student's **last-position** logits vs ground-truth, raw
(un-temperatured) logits:
```
L_rec = (1/B) Σ_b  −log softmax(z^S_b)_{y_b}
```
`z^S = matmul(student_out[:,-1,:], item_embeddings.weight.T)` → [B, |V|] (tied output).

## L_pred — prediction-level KD  (trainers.py:509-512)
KL(p_teacher ‖ p_student) at temperature T, ×T², `batchmean`, last position, full vocab:
```python
log_p_s = F.log_softmax(student_logits / T, dim=-1)
p_t     = F.softmax(teacher_logits / T, dim=-1)
l_pred  = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T*T)
```
```
L_pred = T² · (1/B) Σ_b KL( softmax(z^T_b/T) ‖ softmax(z^S_b/T) )
```
Teacher logits detached (no_grad). Args: λ_pred = `--lambda_kd` (grid {0.5,1.0,2.0}),
T = `--kd_temperature` (grid {1.0,2.0,5.0}).

## L_hs — hidden-state KD  (trainers.py:514-528, HiddenStateKDLoss 191-223)
Aligns **student last-block Mamba-branch output** with **teacher last-block pre-residual
attention output**, MSE over all valid positions, NO projection (D_s=D_t=64):
```
ℓ = N−1 (final block);  m_{i,p} = 1[input_id_{i,p} > 0]
L_hs = [ Σ_{i,p} m_{i,p} · (1/D) Σ_d (h^S_{i,p,d} − h^T_{i,p,d})² ] / [ Σ_{i,p} m_{i,p} + 1e-10 ]
```
- teacher h^T = `block.layer.attention_layer.dense` output (pre-residual, _modules.py:136), detached.
- student h^S = `block.layer.mamba` output (kd_student_v3.py:148), gradient flows.
Args (v3 fixed): λ_hs = `--lambda_hs` (grid {0.05,0.1,0.2}), `hs_loss_type=mse`,
`hs_position_mode=all`, `hs_layer_mode=last`, `hs_use_projection=False`.

## L_cdd — Context-Direction Decorrelation  (trainers.py:226-284) — OFF in v3
δ = h_pre − h_post (last position); L_align = −mean(cos²(δ̂, ŝ)); L_uniform (Wang&Isola);
L_CDD = α·L_align + (1−α)·L_uniform. Never enabled (λ_cdd=0 default, grid never passes it);
`cdd_loss_fn=None`, the teacher `post_{i}` hook is captured but unused.

## Effective v3 setting — USED vs AVAILABLE-BUT-UNUSED

| Component / knob | v3 grid value | Status |
|---|---|---|
| L_rec (full-vocab CE, last pos) | on | USED |
| L_pred (T²·KL teacher‖student) | λ_pred∈{0.5,1,2}, T∈{1,2,5} | USED |
| L_hs (MSE, last block, all positions, no proj) | λ_hs∈{0.05,0.1,0.2} | USED |
| HS loss_type = cosine | — | **AVAILABLE, UNUSED** |
| HS position_mode = last | — | **AVAILABLE, UNUSED** |
| HS layer_mode = all (per-block avg) | — | **AVAILABLE, UNUSED** |
| HS projection (Linear d_s→d_t) | — | **AVAILABLE, UNUSED** (needed for d_s≠d_t; has bug B1/B2) |
| L_cdd | λ_cdd=0 | **AVAILABLE, OFF** |

Empirical (V3_RESULTS): HS-KD contributes little (Pred-only ≈ HS-on); gains are
architecture-driven. So L_pred is the workhorse; L_hs is currently near-inert.

## Hook map (per block i)

| Side | module | detach | consumer |
|---|---|---|---|
| Teacher | `block.layer.attention_layer.dense` (pre-residual) | True | **L_hs** |
| Teacher | `block.layer.attention_layer` (post-residual+LN) | True | L_cdd only (off) |
| Student | `block.layer.mamba` (= `hs_hook_target`) | False | **L_hs** |
