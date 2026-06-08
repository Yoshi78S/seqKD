"""Complementarity diagnostic for adaptive ranking distillation v2.

Tests the COMPLEMENTARITY hypothesis (not standalone z_set performance):
  z_ord (normal teacher) is strong but over-relies on the last item, so it
  misses some ground-truth items. Does z_set (last-item-independent, pre-residual
  signal) catch what z_ord misses? Measured teacher-only, no training, no_grad.
  filter-seen applied (so z_ord hit@10 == the reported HR@10).

For each dataset, for z_set route1 (final-block residual skipped) and route2
(h_pre direct dot), vs the SHARED z_ord (normal teacher):

  Indicator 1  4-way hit classification (K=10, 20): both_hit / ord_only /
               set_only* / both_miss. set_only = complementary potential.
  Indicator 2  Union Recall@K = Pr[y in Top_K(z_ord) ∪ Top_K(z_set)] vs z_ord.
  Indicator 3  rho relationship: set_only rho vs overall; rho-tercile table.
  Indicator 4  for set_only, rank of y in z_ord's ranking (how far below K).

PASS (per dataset, K=10): set_only >= ~5% AND union gain >= ~+3% (relative).
MARGINAL: set_only 2-5% or small gain. FAIL: set_only < 2% and union ~= z_ord.

Usage:  python diagnose_complementarity.py [--datasets ML-1M]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from dataset import RecDataset, get_seq_dic, get_rating_matrix
from model.bsarec import BSARecModel

SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = (SCRIPT_DIR / ".." / ".." / "BSARec" / "src" / "output").resolve()

TEACHERS = {
    "Beauty": dict(heads=2, alpha=0.7, c=5, ckpt="BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt"),
    "LastFM": dict(heads=1, alpha=0.9, c=3, ckpt="BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt"),
    "ML-1M":  dict(heads=1, alpha=0.3, c=9, ckpt="BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt"),
}
KS = [10, 20]


def make_args(ds):
    cfg = TEACHERS[ds]
    a = argparse.Namespace(
        data_dir="../../BSARec/src/data/", data_name=ds, model_type="BSARec",
        max_seq_length=50, hidden_size=64, num_hidden_layers=2, hidden_act="gelu",
        num_attention_heads=cfg["heads"], attention_probs_dropout_prob=0.5,
        hidden_dropout_prob=0.5, initializer_range=0.02, alpha=cfg["alpha"], c=cfg["c"],
        batch_size=256, num_workers=2)
    a.same_target_path = os.path.join(a.data_dir, a.data_name + "_same_target.npy")
    return a


class PrePostHooks:
    def __init__(self, teacher):
        self.set_mode = False; self.h_pre = None; self.h_post = None
        attn = teacher.item_encoder.blocks[-1].layer.attention_layer
        attn.dense.register_forward_hook(self._cap_pre)
        attn.register_forward_hook(self._attn_out)
    def _cap_pre(self, _m, _i, out): self.h_pre = out
    def _attn_out(self, _m, _i, out):
        if self.set_mode: return self.h_pre
        self.h_post = out; return None


@torch.no_grad()
def t_logits(teacher, hooks, input_ids, set_mode=False):
    hooks.set_mode = set_mode
    seq = teacher(input_ids)[:, -1, :]
    out = torch.matmul(seq, teacher.item_embeddings.weight.transpose(0, 1))
    hooks.set_mode = False
    return out


def mask_seen(logits, train_matrix, bidx):
    """Replicate repo eval: zero out seen (train) items. Returns numpy [B,V]."""
    rp = logits.cpu().numpy().copy()
    try:
        rp[train_matrix[bidx].toarray() > 0] = 0
    except Exception:
        rp = rp[:, :-1]
        rp[train_matrix[bidx].toarray() > 0] = 0
    return rp


def hits_and_rank(rp, y, k_list):
    """rp: [B,V] numpy (seen-masked). y: [B] numpy. Returns dict k->hit[B] bool,
    and rank_of_y[B] (1-based, # items strictly greater + 1)."""
    B = rp.shape[0]
    yscore = rp[np.arange(B), y]
    rank = (rp > yscore[:, None]).sum(1) + 1            # [B]
    hits = {}
    for k in k_list:
        # top-k membership: y is in top-k iff rank <= k
        hits[k] = rank <= k
    return hits, rank


def diagnose(ds, device):
    args = make_args(ds)
    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1; args.num_users = num_users + 1
    _, test_rm = get_rating_matrix(args.data_name, seq_dic, max_item)
    test_ds = RecDataset(args, seq_dic["user_seq"], data_type="test")
    loader = DataLoader(test_ds, sampler=SequentialSampler(test_ds),
                        batch_size=args.batch_size, num_workers=args.num_workers)
    teacher = BSARecModel(args)
    teacher.load_state_dict(torch.load(CKPT_DIR / TEACHERS[ds]["ckpt"],
                                       map_location=device, weights_only=False))
    teacher.to(device).eval()
    hooks = PrePostHooks(teacher)
    E = teacher.item_embeddings.weight

    acc = {"rho": [], "y_rank_ord": [],
           "ord": {k: [] for k in KS},
           "r1": {k: [] for k in KS}, "r2": {k: [] for k in KS}}
    with torch.no_grad():
        for batch in loader:
            user_ids, input_ids, answers, _, _ = (t.to(device) for t in batch)
            bidx = user_ids.cpu().numpy(); y = answers.cpu().numpy()
            z_ord = t_logits(teacher, hooks, input_ids, set_mode=False)
            h_pre = hooks.h_pre[:, -1, :]; h_post = hooks.h_post[:, -1, :]
            z_r1 = t_logits(teacher, hooks, input_ids, set_mode=True)
            z_r2 = torch.matmul(h_pre, E.transpose(0, 1))

            acc["rho"].append((1 - F.cosine_similarity(h_pre, h_post, dim=-1)).clamp(0, 1).cpu().numpy())

            rp_ord = mask_seen(z_ord, test_rm, bidx)
            rp_r1 = mask_seen(z_r1, test_rm, bidx)
            rp_r2 = mask_seen(z_r2, test_rm, bidx)
            ord_hits, ord_rank = hits_and_rank(rp_ord, y, KS)
            r1_hits, _ = hits_and_rank(rp_r1, y, KS)
            r2_hits, _ = hits_and_rank(rp_r2, y, KS)
            acc["y_rank_ord"].append(ord_rank)
            for k in KS:
                acc["ord"][k].append(ord_hits[k]); acc["r1"][k].append(r1_hits[k]); acc["r2"][k].append(r2_hits[k])

    rho = np.concatenate(acc["rho"])
    y_rank_ord = np.concatenate(acc["y_rank_ord"])
    ord_h = {k: np.concatenate(acc["ord"][k]) for k in KS}
    r1_h = {k: np.concatenate(acc["r1"][k]) for k in KS}
    r2_h = {k: np.concatenate(acc["r2"][k]) for k in KS}
    return rho, y_rank_ord, ord_h, {"route1": r1_h, "route2": r2_h}


def report_route(ds, route, rho, y_rank_ord, ord_h, set_h):
    print(f"\n----- {ds} / {route} -----")
    N = len(rho)
    for k in KS:
        o = ord_h[k]; s = set_h[k]
        both = (o & s).mean(); ord_only = (o & ~s).mean()
        set_only = (~o & s).mean(); both_miss = (~o & ~s).mean()
        union = (o | s).mean(); ordr = o.mean()
        gain = union - ordr; gain_rel = 100 * gain / max(ordr, 1e-9)
        print(f"  K={k:2d}: both={both*100:4.1f}%  ord_only={ord_only*100:4.1f}%  "
              f"**set_only={set_only*100:4.1f}%**  both_miss={both_miss*100:4.1f}%")
        print(f"        Recall@{k}: z_ord={ordr*100:5.2f}%  union={union*100:5.2f}%  "
              f"gain=+{gain*100:.2f}pt (+{gain_rel:.1f}% rel)")
    # Indicator 3: rho relationship (K=10)
    so10 = (~ord_h[10] & set_h[10])
    print(f"  [rho] set_only rho_mean={rho[so10].mean() if so10.any() else float('nan'):.3f} "
          f"vs overall={rho.mean():.3f}")
    q1, q2 = np.quantile(rho, [1/3, 2/3])
    terc = [("low", rho <= q1), ("mid", (rho > q1) & (rho <= q2)), ("high", rho > q2)]
    print(f"  [rho-tercile]  set_only@10 / union-gain@10:")
    for name, m in terc:
        if m.sum() == 0: continue
        so = (~ord_h[10][m] & set_h[10][m]).mean()
        g = ((ord_h[10][m] | set_h[10][m]).mean() - ord_h[10][m].mean()) * 100
        print(f"     {name:4s} (rho<= {rho[m].max():.2f}, n={m.sum():5d}): "
              f"set_only={so*100:4.1f}%  union-gain=+{g:.2f}pt")
    # Indicator 4: rank of y in z_ord for set_only samples (K=10)
    if so10.any():
        rk = y_rank_ord[so10]
        within2k = (rk <= 20).mean() * 100
        within10k = ((rk > 20) & (rk <= 100)).mean() * 100
        far = (rk > 100).mean() * 100
        print(f"  [quality] set_only y-rank in z_ord: median={np.median(rk):.0f} mean={rk.mean():.0f} | "
              f"<=2K(20):{within2k:.0f}%  20-100:{within10k:.0f}%  >100:{far:.0f}%")
    # verdict (K=10)
    so = (~ord_h[10] & set_h[10]).mean() * 100
    gain_rel = 100 * ((ord_h[10] | set_h[10]).mean() - ord_h[10].mean()) / max(ord_h[10].mean(), 1e-9)
    if so >= 5 and gain_rel >= 3:
        v = "PASS (complementary)"
    elif so >= 2:
        v = "MARGINAL"
    else:
        v = "FAIL (no complement)"
    print(f"  => VERDICT (K=10): set_only={so:.1f}%, union-gain=+{gain_rel:.1f}% rel  ->  {v}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["Beauty", "LastFM", "ML-1M"])
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    for ds in args.datasets:
        print(f"\n{'='*64}\n{ds}  (teacher alpha={TEACHERS[ds]['alpha']})\n{'='*64}")
        rho, y_rank_ord, ord_h, sets = diagnose(ds, device)
        for route in ["route1", "route2"]:
            report_route(ds, route, rho, y_rank_ord, ord_h, sets[route])


if __name__ == "__main__":
    main()
