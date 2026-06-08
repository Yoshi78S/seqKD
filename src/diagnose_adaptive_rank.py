"""Pre-flight diagnostics for adaptive ranking distillation (no training).

Runs ONLY teacher forward passes (BSARec) with attention-temperature
manipulation and reports, per dataset:

  A. Divergence between the order target (z_ord, sharpened attn) and the set
     target (z_set, flattened attn): KL(softmax(z_ord) ‖ softmax(z_set)) and
     Top-K overlap. If ≈0/≈1 the adaptive interpolation can't help (concern 1).
  B. HR@10 / NDCG@10 of z_set ALONE (and z_ord, baseline teacher) under the
     standard filter-seen protocol — is tau_set flattening "set emphasis" or
     just degradation? Sweeps several tau_set values.
  C. Histogram of per-sample rho (pre-residual last-position attention entropy)
     over the test set — bimodal (distinct order/set populations) or narrow?

Usage (from seqKD/src/):
  python diagnose_adaptive_rank.py
  python diagnose_adaptive_rank.py --datasets ML-1M --tau_set 1.3 1.5 2.0
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from dataset import RecDataset, get_seq_dic, get_rating_matrix
from model.bsarec import BSARecModel
from metrics import recall_at_k, ndcg_k

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
CKPT_DIR = (SCRIPT_DIR / ".." / ".." / "BSARec" / "src" / "output").resolve()
RESULT_DIR = SCRIPT_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

TEACHERS = {
    "Beauty": dict(heads=2, alpha=0.7, c=5,
                   ckpt="BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt"),
    "LastFM": dict(heads=1, alpha=0.9, c=3,
                   ckpt="BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt"),
    "ML-1M":  dict(heads=1, alpha=0.3, c=9,
                   ckpt="BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt"),
}


def make_args(ds):
    cfg = TEACHERS[ds]
    a = argparse.Namespace(
        data_dir="../../BSARec/src/data/", data_name=ds, output_dir=str(OUTPUT_DIR),
        model_type="BSARec", max_seq_length=50, hidden_size=64, num_hidden_layers=2,
        hidden_act="gelu", num_attention_heads=cfg["heads"],
        attention_probs_dropout_prob=0.5, hidden_dropout_prob=0.5,
        initializer_range=0.02, alpha=cfg["alpha"], c=cfg["c"],
        batch_size=256, num_workers=2, no_cuda=False, seed=42)
    a.same_target_path = os.path.join(a.data_dir, a.data_name + "_same_target.npy")
    return a


class AttnTempHooks:
    """forward_pre_hook on each teacher attention softmax to divide scores by
    tau; forward_hook to capture probs (for rho)."""
    def __init__(self, teacher):
        self.tau = 1.0
        self.capture = False
        self.probs = {}
        self._hooks = []
        for i, blk in enumerate(teacher.item_encoder.blocks):
            sm = blk.layer.attention_layer.softmax
            self._hooks.append(sm.register_forward_pre_hook(self._pre))
            self._hooks.append(sm.register_forward_hook(self._mk_cap(i)))

    def _pre(self, _m, inp):
        if self.tau == 1.0:
            return None
        return (inp[0] / self.tau,) + tuple(inp[1:])

    def _mk_cap(self, idx):
        def hook(_m, _i, out):
            if self.capture:
                self.probs[idx] = out.detach()
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


@torch.no_grad()
def teacher_logits(teacher, hooks, input_ids, tau, capture=False):
    hooks.tau = tau
    hooks.capture = capture
    if capture:
        hooks.probs = {}
    seq_out = teacher(input_ids)[:, -1, :]
    logits = torch.matmul(seq_out, teacher.item_embeddings.weight.transpose(0, 1))
    hooks.tau = 1.0
    hooks.capture = False
    return logits


def compute_rho(hooks, input_ids):
    valid_len = (input_ids > 0).sum(dim=1)
    log_len = torch.log(valid_len.float().clamp(min=2.0))
    rhos = []
    for probs in hooks.probs.values():
        a = probs[:, :, -1, :].clamp_min(1e-12)
        ent = -(a * a.log()).sum(dim=-1)
        rhos.append((1.0 - ent / log_len[:, None]).mean(dim=1))
    rho = torch.stack(rhos, dim=0).mean(dim=0).clamp(0.0, 1.0)
    return torch.where(valid_len <= 1, torch.ones_like(rho), rho)


def topk_overlap(z_a, z_b, k):
    ta = torch.topk(z_a, k, dim=1).indices
    tb = torch.topk(z_b, k, dim=1).indices
    # overlap count per row
    out = []
    for i in range(ta.shape[0]):
        out.append(len(set(ta[i].tolist()) & set(tb[i].tolist())) / k)
    return torch.tensor(out, device=z_a.device)


def score_hr(rating_pred_np, answers_np, train_matrix, batch_user_index):
    """Filter-seen + top-20 → per-batch pred list (item ids)."""
    rp = rating_pred_np.copy()
    try:
        rp[train_matrix[batch_user_index].toarray() > 0] = 0
    except Exception:
        rp = rp[:, :-1]
        rp[train_matrix[batch_user_index].toarray() > 0] = 0
    ind = np.argpartition(rp, -20)[:, -20:]
    arr = rp[np.arange(len(rp))[:, None], ind]
    arr_argsort = np.argsort(arr)[np.arange(len(rp)), ::-1]
    return ind[np.arange(len(rp))[:, None], arr_argsort]


def diagnose(ds, tau_ord, tau_sets, topk_k, device):
    args = make_args(ds)
    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1
    _, test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    test_ds = RecDataset(args, seq_dic["user_seq"], data_type="test")
    loader = DataLoader(test_ds, sampler=SequentialSampler(test_ds),
                        batch_size=args.batch_size, num_workers=args.num_workers)

    teacher = BSARecModel(args)
    teacher.load_state_dict(torch.load(CKPT_DIR / TEACHERS[ds]["ckpt"],
                                       map_location=device, weights_only=False))
    teacher.to(device).eval()
    hooks = AttnTempHooks(teacher)

    rho_all = []
    kl_ord_set = {ts: [] for ts in tau_sets}
    ov10 = {ts: [] for ts in tau_sets}
    ovK = {ts: [] for ts in tau_sets}
    # pred lists for HR: baseline, ord, each set
    preds = {"base": [], "ord": [], **{f"set{ts}": [] for ts in tau_sets}}
    answers_all = []

    try:
        for batch in loader:
            user_ids, input_ids, answers, _, _ = (t.to(device) for t in batch)
            bidx = user_ids.cpu().numpy()
            ans_np = answers.cpu().numpy()

            z_base = teacher_logits(teacher, hooks, input_ids, 1.0, capture=True)
            rho_all.append(compute_rho(hooks, input_ids).cpu())
            z_ord = teacher_logits(teacher, hooks, input_ids, tau_ord)

            preds["base"].append(score_hr(z_base.cpu().numpy(), ans_np, test_rating_matrix, bidx))
            preds["ord"].append(score_hr(z_ord.cpu().numpy(), ans_np, test_rating_matrix, bidx))
            answers_all.append(ans_np)

            lp_ord = torch.log_softmax(z_ord, dim=-1)
            for ts in tau_sets:
                z_set = teacher_logits(teacher, hooks, input_ids, ts)
                # KL(softmax(z_ord) || softmax(z_set))
                p_ord = lp_ord.exp()
                kl = (p_ord * (lp_ord - torch.log_softmax(z_set, dim=-1))).sum(dim=-1)
                kl_ord_set[ts].append(kl.cpu())
                ov10[ts].append(topk_overlap(z_ord, z_set, 10).cpu())
                ovK[ts].append(topk_overlap(z_ord, z_set, topk_k).cpu())
                preds[f"set{ts}"].append(
                    score_hr(z_set.cpu().numpy(), ans_np, test_rating_matrix, bidx))
    finally:
        hooks.remove()

    rho = torch.cat(rho_all).numpy()
    answers_np = np.concatenate(answers_all)

    def hr(name):
        pl = np.concatenate(preds[name], axis=0)
        return recall_at_k(answers_np.tolist(), pl, 10), ndcg_k(answers_np.tolist(), pl, 10)

    res = {
        "rho": rho,
        "hr_base": hr("base"),
        "hr_ord": hr("ord"),
        "hr_set": {ts: hr(f"set{ts}") for ts in tau_sets},
        "kl": {ts: float(torch.cat(kl_ord_set[ts]).mean()) for ts in tau_sets},
        "ov10": {ts: float(torch.cat(ov10[ts]).mean()) for ts in tau_sets},
        "ovK": {ts: float(torch.cat(ovK[ts]).mean()) for ts in tau_sets},
    }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["Beauty", "LastFM", "ML-1M"])
    p.add_argument("--tau_ord", type=float, default=0.5)
    p.add_argument("--tau_set", nargs="+", type=float, default=[1.3, 1.5, 2.0])
    p.add_argument("--topk_k", type=int, default=50)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")

    all_res = {}
    for ds in args.datasets:
        print(f"\n{'='*60}\n{ds}  (teacher alpha={TEACHERS[ds]['alpha']}, "
              f"attn weight 1-alpha={1-TEACHERS[ds]['alpha']:.2f})\n{'='*60}")
        r = diagnose(ds, args.tau_ord, args.tau_set, args.topk_k, device)
        all_res[ds] = r

        print(f"\n[C] rho histogram (N={len(r['rho'])}, "
              f"mean={r['rho'].mean():.3f}, std={r['rho'].std():.3f}):")
        hist, edges = np.histogram(r["rho"], bins=10, range=(0, 1))
        for j in range(10):
            bar = "#" * int(50 * hist[j] / max(hist.max(), 1))
            print(f"  [{edges[j]:.1f},{edges[j+1]:.1f}) {hist[j]:6d} {bar}")

        print(f"\n[B] HR@10 / NDCG@10 (filter-seen):")
        print(f"  baseline teacher (tau=1.0) : {r['hr_base'][0]:.4f} / {r['hr_base'][1]:.4f}")
        print(f"  z_ord (tau={args.tau_ord})       : {r['hr_ord'][0]:.4f} / {r['hr_ord'][1]:.4f}")
        for ts in args.tau_set:
            h = r["hr_set"][ts]
            drop = 100 * (h[0] - r["hr_base"][0]) / r["hr_base"][0]
            print(f"  z_set (tau={ts})        : {h[0]:.4f} / {h[1]:.4f}  ({drop:+.1f}% vs base)")

        print(f"\n[A] ord vs set divergence (tau_ord={args.tau_ord}):")
        print(f"  {'tau_set':>8} | {'KL(ord‖set)':>12} | {'overlap@10':>10} | {'overlap@'+str(args.topk_k):>11}")
        for ts in args.tau_set:
            print(f"  {ts:>8} | {r['kl'][ts]:>12.4f} | {r['ov10'][ts]:>10.3f} | {r['ovK'][ts]:>11.3f}")

    # rho histogram figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(all_res), figsize=(5 * len(all_res), 4),
                                 squeeze=False)
        for ax, (ds, r) in zip(axes[0], all_res.items()):
            ax.hist(r["rho"], bins=30, range=(0, 1), color="#1f78b4", edgecolor="black", lw=0.3)
            ax.set_title(f"{ds}  (mean ρ={r['rho'].mean():.2f})")
            ax.set_xlabel("ρ (order-dependence)")
            ax.set_ylabel("count")
        fig.suptitle("Per-sample order-dependence ρ (pre-residual attention entropy)")
        fig.tight_layout()
        out = RESULT_DIR / "adaptive_rank_rho_hist.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved rho histogram figure: {out}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
