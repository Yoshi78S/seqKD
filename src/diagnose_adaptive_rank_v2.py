"""Gating diagnostics for adaptive ranking distillation v2 (pre/post-residual).

Teacher-only (no training). Per dataset reports:

  A'  z_ord vs z_set divergence: KL(z_ord ‖ z_set) + Top-10/Top-50 overlap,
      for route1 (final-block residual skipped) and route2 (direct dot).
      PASS if Top-10 overlap clearly < the v1 failure band (78-92%); ~<0.6 good.
  B'  z_set ALONE HR@10/NDCG@10 (filter-seen) vs the baseline teacher (z_ord).
      PASS if z_set is within ~-10% of baseline (not collapsed / OOD).
  C'  rho = 1 - cos(h_pre_last, h_post_last): histogram, mean/std, and the
      per-sample correlation with HRLI@1. PASS if rho is broadly spread
      (ideally bimodal) and positively correlated with HRLI@1.

Usage (from seqKD/src/):
  python diagnose_adaptive_rank_v2.py
  python diagnose_adaptive_rank_v2.py --datasets ML-1M
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F

from dataset import RecDataset, get_seq_dic, get_rating_matrix
from model.bsarec import BSARecModel
from metrics import recall_at_k, ndcg_k

SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = (SCRIPT_DIR / ".." / ".." / "BSARec" / "src" / "output").resolve()
RESULT_DIR = SCRIPT_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

TEACHERS = {
    "Beauty": dict(heads=2, alpha=0.7, c=5, ckpt="BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt"),
    "LastFM": dict(heads=1, alpha=0.9, c=3, ckpt="BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt"),
    "ML-1M":  dict(heads=1, alpha=0.3, c=9, ckpt="BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt"),
}


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
        self.set_mode = False
        self.h_pre = None
        self.h_post = None
        attn = teacher.item_encoder.blocks[-1].layer.attention_layer
        attn.dense.register_forward_hook(self._cap_pre)
        attn.register_forward_hook(self._attn_out)

    def _cap_pre(self, _m, _i, out):
        self.h_pre = out

    def _attn_out(self, _m, _i, out):
        if self.set_mode:
            return self.h_pre
        self.h_post = out
        return None


@torch.no_grad()
def t_logits(teacher, hooks, input_ids, set_mode=False):
    hooks.set_mode = set_mode
    seq = teacher(input_ids)[:, -1, :]
    logits = torch.matmul(seq, teacher.item_embeddings.weight.transpose(0, 1))
    hooks.set_mode = False
    return logits


def topk_overlap(a, b, k):
    ta = torch.topk(a, k, dim=1).indices
    tb = torch.topk(b, k, dim=1).indices
    return torch.tensor([len(set(ta[i].tolist()) & set(tb[i].tolist())) / k
                         for i in range(ta.shape[0])])


def kl_rows(z_p, z_q):
    lp = F.log_softmax(z_p, -1)
    lq = F.log_softmax(z_q, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def score_hr(rp_np, train_matrix, bidx):
    rp = rp_np.copy()
    try:
        rp[train_matrix[bidx].toarray() > 0] = 0
    except Exception:
        rp = rp[:, :-1]
        rp[train_matrix[bidx].toarray() > 0] = 0
    ind = np.argpartition(rp, -20)[:, -20:]
    arr = rp[np.arange(len(rp))[:, None], ind]
    asort = np.argsort(arr)[np.arange(len(rp)), ::-1]
    return ind[np.arange(len(rp))[:, None], asort]


def diagnose(ds, device):
    args = make_args(ds)
    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1
    _, test_rm = get_rating_matrix(args.data_name, seq_dic, max_item)
    test_ds = RecDataset(args, seq_dic["user_seq"], data_type="test")
    loader = DataLoader(test_ds, sampler=SequentialSampler(test_ds),
                        batch_size=args.batch_size, num_workers=args.num_workers)
    teacher = BSARecModel(args)
    teacher.load_state_dict(torch.load(CKPT_DIR / TEACHERS[ds]["ckpt"],
                                       map_location=device, weights_only=False))
    teacher.to(device).eval()
    hooks = PrePostHooks(teacher)

    rho_all, hrli_all, ans_all = [], [], []
    kl_r1, kl_r2, ov10_r1, ov50_r1, ov10_r2, ov50_r2 = ([] for _ in range(6))
    preds = {k: [] for k in ["ord", "set_r1", "set_r2", "ord_r2"]}
    E = teacher.item_embeddings.weight

    for batch in loader:
        user_ids, input_ids, answers, _, _ = (t.to(device) for t in batch)
        bidx = user_ids.cpu().numpy()
        z_ord = t_logits(teacher, hooks, input_ids, set_mode=False)
        h_pre = hooks.h_pre[:, -1, :]
        h_post = hooks.h_post[:, -1, :]
        z_set_r1 = t_logits(teacher, hooks, input_ids, set_mode=True)
        # E (teacher emb) requires grad; detach the route2 direct-dot logits.
        z_set_r2 = torch.matmul(h_pre, E.transpose(0, 1)).detach()
        z_ord_r2 = torch.matmul(h_post, E.transpose(0, 1)).detach()

        rho_all.append((1 - F.cosine_similarity(h_pre, h_post, dim=-1)).clamp(0, 1).cpu())
        last_item = input_ids[:, -1]
        hrli_all.append((z_ord.argmax(-1) == last_item).float().cpu())
        ans_all.append(answers.cpu().numpy())

        kl_r1.append(kl_rows(z_ord, z_set_r1).cpu())
        kl_r2.append(kl_rows(z_ord_r2, z_set_r2).cpu())
        ov10_r1.append(topk_overlap(z_ord, z_set_r1, 10)); ov50_r1.append(topk_overlap(z_ord, z_set_r1, 50))
        ov10_r2.append(topk_overlap(z_ord_r2, z_set_r2, 10)); ov50_r2.append(topk_overlap(z_ord_r2, z_set_r2, 50))

        preds["ord"].append(score_hr(z_ord.cpu().numpy(), test_rm, bidx))
        preds["set_r1"].append(score_hr(z_set_r1.cpu().numpy(), test_rm, bidx))
        preds["set_r2"].append(score_hr(z_set_r2.cpu().numpy(), test_rm, bidx))
        preds["ord_r2"].append(score_hr(z_ord_r2.cpu().numpy(), test_rm, bidx))

    rho = torch.cat(rho_all).numpy()
    hrli = torch.cat(hrli_all).numpy()
    ans = np.concatenate(ans_all).tolist()

    def hr(name):
        pl = np.concatenate(preds[name], axis=0)
        return recall_at_k(ans, pl, 10), ndcg_k(ans, pl, 10)

    return {
        "rho": rho, "hrli": hrli,
        "corr": float(np.corrcoef(rho, hrli)[0, 1]),
        "kl_r1": float(torch.cat(kl_r1).mean()), "kl_r2": float(torch.cat(kl_r2).mean()),
        "ov10_r1": float(torch.cat(ov10_r1).mean()), "ov50_r1": float(torch.cat(ov50_r1).mean()),
        "ov10_r2": float(torch.cat(ov10_r2).mean()), "ov50_r2": float(torch.cat(ov50_r2).mean()),
        "hr": {k: hr(k) for k in preds},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["Beauty", "LastFM", "ML-1M"])
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")

    all_res = {}
    for ds in args.datasets:
        print(f"\n{'='*64}\n{ds}  (teacher alpha={TEACHERS[ds]['alpha']})\n{'='*64}")
        r = diagnose(ds, device)
        all_res[ds] = r

        print("[A'] z_ord vs z_set divergence  (PASS if Top-10 overlap << 0.78, ~<0.6)")
        print(f"   route1: KL={r['kl_r1']:.4f}  overlap@10={r['ov10_r1']:.3f}  overlap@50={r['ov50_r1']:.3f}")
        print(f"   route2: KL={r['kl_r2']:.4f}  overlap@10={r['ov10_r2']:.3f}  overlap@50={r['ov50_r2']:.3f}")

        b = r["hr"]
        print("\n[B'] z_set HR@10 / NDCG@10 (filter-seen)  (PASS if z_set within ~-10% of baseline)")
        print(f"   baseline z_ord (normal)   : {b['ord'][0]:.4f} / {b['ord'][1]:.4f}")
        d1 = 100*(b['set_r1'][0]-b['ord'][0])/b['ord'][0]
        d2 = 100*(b['set_r2'][0]-b['ord'][0])/b['ord'][0]
        print(f"   z_set route1              : {b['set_r1'][0]:.4f} / {b['set_r1'][1]:.4f}  ({d1:+.1f}% vs base)")
        print(f"   z_set route2              : {b['set_r2'][0]:.4f} / {b['set_r2'][1]:.4f}  ({d2:+.1f}% vs base)")
        print(f"   z_ord route2 (h_post dot) : {b['ord_r2'][0]:.4f} / {b['ord_r2'][1]:.4f}")

        print(f"\n[C'] rho = 1-cos(h_pre,h_post)  mean={r['rho'].mean():.3f} std={r['rho'].std():.3f}  "
              f"corr(rho, HRLI@1)={r['corr']:+.3f}  (PASS if spread + positive corr)")
        hist, edges = np.histogram(r["rho"], bins=10, range=(0, 1))
        for j in range(10):
            bar = "#" * int(50 * hist[j] / max(hist.max(), 1))
            print(f"   [{edges[j]:.1f},{edges[j+1]:.1f}) {hist[j]:6d} {bar}")

    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(all_res), figsize=(5*len(all_res), 4), squeeze=False)
        for a, (ds, r) in zip(ax[0], all_res.items()):
            a.hist(r["rho"], bins=30, range=(0, 1), color="#e31a1c", edgecolor="black", lw=0.3)
            a.set_title(f"{ds}  mean ρ={r['rho'].mean():.2f}, corr={r['corr']:+.2f}")
            a.set_xlabel("ρ = 1 - cos(h_pre, h_post)"); a.set_ylabel("count")
        fig.suptitle("v2 order-dependence ρ (pre/post-residual divergence)")
        fig.tight_layout()
        out = RESULT_DIR / "adaptive_rank_v2_rho_hist.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
