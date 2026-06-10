"""Offline diagnostics D0-D4 for kappa-corrected relational distillation.

kappa = d * (1 - a), where (teacher quantities, frozen ckpt):
  d = cos(h_t, E_t[ell]).clamp(0,1)   -- teacher's per-instance dominance
  a = cos(E_t[ell], E_t[y]).clamp(0,1) -- correctness of the ell-neighborhood strategy
g = h_t - min(kappa,0.95) * (h_t . e_l) * e_l   -- rank-1 soft removal

D1: where does PL-KD help/hurt? KD vs no-KD student HR@10 by test-kappa tercile.
D2: does kappa predict teacher failure? (valid) teacher HR@10 / mean rank(y) by
    tercile of d, (1-a), kappa; Spearman correlations.
D3: is the corrected g a sane relational target? (valid) score by h_t.E^T vs
    g.E^T, HR@10/rank by kappa tercile.  PASS: high-kappa HR_g > HR_h, low-kappa ~=.
D4: contamination of teacher geometry: 1e5 random valid pairs, corr of
    cos(h_i,h_j) vs cos(El_i,El_j) and vs cos(Ey_i,Ey_j), before/after correction.
Degeneracy: fraction ||g|| < 0.1||h_t|| (train+valid; ell==0 train rows excluded).

Run from seqKD/src/:  python diag_kappa.py [--datasets ML-1M Beauty LastFM]
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

CKPT = "../../BSARec/src/output"
TEACH = {
    "Beauty": dict(ckpt="BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt", heads="2", alpha="0.7", c="5", drop="0.5"),
    "LastFM": dict(ckpt="BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt",  heads="1", alpha="0.9", c="3", drop="0.5"),
    "ML-1M":  dict(ckpt="BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt",  heads="1", alpha="0.3", c="9", drop="0.2"),
}
STUDENT_KD = {ds: f"cmi_linear_{ds}_b0" for ds in TEACH}      # best PL-KD students
STUDENT_NOKD = {ds: f"fmamba_noKD_{ds}" for ds in TEACH}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(ds):
    from utils import parse_args
    from dataset import get_seq_dic, get_dataloder, get_rating_matrix
    cfg = TEACH[ds]
    sys.argv = ["diag", "--model_type", "kdstudent_v3", "--data_name", ds, "--train_name", "diag",
                "--alpha", cfg["alpha"], "--c", cfg["c"], "--d_state", "16", "--d_conv", "4",
                "--expand", "1", "--hidden_size", "64", "--num_hidden_layers", "2",
                "--hidden_dropout_prob", cfg["drop"], "--attention_probs_dropout_prob", cfg["drop"],
                "--batch_size", "256", "--seed", "42"]
    args = parse_args()
    args.cuda_condition = DEVICE.type == "cuda"
    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1
    args.same_target_path = os.path.join(args.data_dir, ds + "_same_target.npy")
    loaders = get_dataloder(args, seq_dic)
    vrm, trm = get_rating_matrix(ds, seq_dic, max_item)
    return args, loaders, vrm, trm


def load_teacher(ds, args):
    import argparse as ap
    from model import MODEL_DICT
    cfg = TEACH[ds]
    ta = ap.Namespace(**vars(args))
    ta.num_attention_heads = int(cfg["heads"])
    ta.alpha = float(cfg["alpha"]); ta.c = int(cfg["c"])
    t = MODEL_DICT["bsarec"](args=ta)
    t.load_state_dict(torch.load(os.path.join(CKPT, cfg["ckpt"]), map_location=DEVICE))
    return t.to(DEVICE).eval()


def load_student(name, args):
    from model import MODEL_DICT
    m = MODEL_DICT["kdstudent_v3"](args=args)
    m.load_state_dict(torch.load(f"output/{name}.pt", map_location=DEVICE))
    return m.to(DEVICE).eval()


@torch.no_grad()
def teacher_quant(teacher, input_ids, answers):
    """h_t, kappa, g for one batch (the on-the-fly computation used in training)."""
    E = teacher.item_embeddings.weight
    h_t = teacher.predict(input_ids, None)[:, -1, :]
    ell = input_ids[:, -1]
    e_l = F.normalize(E[ell], dim=-1)
    e_y = F.normalize(E[answers], dim=-1)
    d = (F.normalize(h_t, dim=-1) * e_l).sum(-1).clamp(0, 1)
    a = (e_l * e_y).sum(-1).clamp(0, 1)
    kappa = d * (1 - a)
    valid = (ell != 0).float()
    kappa = kappa * valid                                  # pad-last rows: no correction
    p = (h_t * e_l).sum(-1, keepdim=True)
    c = kappa.clamp(max=0.95).unsqueeze(-1)
    g = h_t - c * p * e_l
    return h_t, kappa, g, d, a, ell


@torch.no_grad()
def ranks_hr(scores, answers, user_idx, rating_matrix):
    """filter-seen rank of y (repo-style) -> (rank array, hit@10)."""
    rp = scores.cpu().numpy().copy()
    y = answers.cpu().numpy()
    try:
        rp[rating_matrix[user_idx].toarray() > 0] = 0
    except Exception:
        rp = rp[:, :-1]; rp[rating_matrix[user_idx].toarray() > 0] = 0
    rk = (rp > rp[np.arange(len(rp)), y][:, None]).sum(1) + 1
    return rk, (rk <= 10)


def terciles(x):
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    return x <= q1, (x > q1) & (x <= q2), x > q2


@torch.no_grad()
def run_ds(ds):
    print(f"\n{'='*78}\n{ds}\n{'='*78}")
    args, (train_l, valid_l, test_l), vrm, trm = build(ds)
    teacher = load_teacher(ds, args)
    E_t = teacher.item_embeddings.weight

    # ---------- collect valid-split teacher quantities ----------
    V = {k: [] for k in ("ht", "g", "kap", "d", "a", "el", "ey", "rk_h", "hit_h", "rk_g", "hit_g", "uid")}
    for batch in valid_l:
        batch = tuple(t.to(DEVICE) for t in batch)
        uid, input_ids, answers = batch[0], batch[1], batch[2]
        h_t, kap, g, d, a, ell = teacher_quant(teacher, input_ids, answers)
        bidx = uid.cpu().numpy()
        rk_h, hit_h = ranks_hr(h_t @ E_t.T, answers, bidx, vrm)
        rk_g, hit_g = ranks_hr(g @ E_t.T, answers, bidx, vrm)
        V["ht"].append(F.normalize(h_t, dim=-1).cpu()); V["g"].append(F.normalize(g, dim=-1).cpu())
        V["kap"].append(kap.cpu()); V["d"].append(d.cpu()); V["a"].append(a.cpu())
        V["el"].append(F.normalize(E_t[ell], dim=-1).cpu())
        V["ey"].append(F.normalize(E_t[answers], dim=-1).cpu())
        V["rk_h"].append(rk_h); V["hit_h"].append(hit_h)
        V["rk_g"].append(rk_g); V["hit_g"].append(hit_g); V["uid"].append(bidx)
    for k in V:
        V[k] = np.concatenate(V[k]) if isinstance(V[k][0], np.ndarray) else torch.cat(V[k]).numpy()
    kap = V["kap"]

    # ---------- degeneracy check (valid + train) ----------
    def degen(loader, name, limit=None):
        n_bad = n = 0
        for bi, batch in enumerate(loader):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, answers = batch[1], batch[2]
            h_t, _, g, _, _, ell = teacher_quant(teacher, input_ids, answers)
            m = ell != 0
            n_bad += int((g[m].norm(dim=-1) < 0.1 * h_t[m].norm(dim=-1)).sum())
            n += int(m.sum())
            if limit and bi + 1 >= limit:
                break
        print(f"  degeneracy ||g||<0.1||h_t||  [{name}]: {100*n_bad/max(n,1):.2f}%  (N={n})")
        return n_bad / max(n, 1)
    dg_v = degen(valid_l, "valid")
    dg_t = degen(train_l, "train")
    if max(dg_v, dg_t) > 0.05:
        print("  *** DEGENERACY > 5% — STOP AND REPORT (per spec) ***")

    # ---------- D2 ----------
    print(f"\nD2: does kappa predict TEACHER failure? (valid, N={len(kap)})")
    print(f"  {'signal':10s} {'lo HR@10':>9} {'mid':>7} {'hi':>7} | {'lo rank':>8} {'mid':>8} {'hi':>8} | spearman(sig, rank)")
    for nm, sig in (("d", V["d"]), ("1-a", 1 - V["a"]), ("kappa", kap)):
        lo, mid, hi = terciles(sig)
        rho = spearmanr(sig, V["rk_h"]).statistic
        print(f"  {nm:10s} {V['hit_h'][lo].mean():9.4f} {V['hit_h'][mid].mean():7.4f} {V['hit_h'][hi].mean():7.4f} | "
              f"{V['rk_h'][lo].mean():8.1f} {V['rk_h'][mid].mean():8.1f} {V['rk_h'][hi].mean():8.1f} | rho={rho:+.3f}")

    # ---------- D3 ----------
    print(f"\nD3: corrected g as scoring state vs h_t (valid, kappa terciles)")
    lo, mid, hi = terciles(kap)
    res = {}
    for nm, msk in (("lo", lo), ("mid", mid), ("hi", hi)):
        hr_h, hr_g = V["hit_h"][msk].mean(), V["hit_g"][msk].mean()
        res[nm] = (hr_h, hr_g)
        print(f"  kappa-{nm}: HR@10 h_t={hr_h:.4f}  g={hr_g:.4f}  (rank {V['rk_h'][msk].mean():.0f} -> {V['rk_g'][msk].mean():.0f})")
    d3_pass = (res["hi"][1] > res["hi"][0]) and (abs(res["lo"][1] - res["lo"][0]) < 0.01)
    print(f"  D3 verdict: {'PASS (arm A = corrected)' if d3_pass else 'FAIL (arm A -> pairgate)'}")

    # ---------- D4 ----------
    N = len(kap)
    rng = np.random.default_rng(0)
    i = rng.integers(0, N, 100_000); j = rng.integers(0, N, 100_000)
    keep = i != j; i, j = i[keep], j[keep]
    ht = torch.from_numpy(V["ht"]); gg = torch.from_numpy(V["g"])
    el = torch.from_numpy(V["el"]); ey = torch.from_numpy(V["ey"])
    c_h = (ht[i] * ht[j]).sum(-1).numpy(); c_g = (gg[i] * gg[j]).sum(-1).numpy()
    c_l = (el[i] * el[j]).sum(-1).numpy(); c_y = (ey[i] * ey[j]).sum(-1).numpy()
    print(f"\nD4: geometry contamination ({len(i)} valid pairs)")
    rows = []
    for tgt_nm, tgt in (("ell-sim", c_l), ("y-sim", c_y)):
        for rep_nm, rep in (("h_t", c_h), ("g", c_g)):
            rho = spearmanr(rep, tgt).statistic
            r2 = np.corrcoef(rep, tgt)[0, 1] ** 2
            rows.append((tgt_nm, rep_nm, rho, r2))
            print(f"  cos(rep_i,rep_j) vs cos({tgt_nm}): rep={rep_nm:3s}  spearman={rho:+.3f}  R2={r2:.3f}")
    os.makedirs("results", exist_ok=True)
    np.savetxt(f"results/kappa_d4_pairs_{ds}.csv",
               np.stack([c_h, c_g, c_l, c_y], 1)[:20000], delimiter=",",
               header="cos_ht,cos_g,cos_El,cos_Ey", comments="")

    # ---------- D1 ----------
    print(f"\nD1: PL-KD gain by TEST-kappa tercile (KD vs noKD student)")
    kd = load_student(STUDENT_KD[ds], args)
    nk = load_student(STUDENT_NOKD[ds], args)
    kap_te, hit_kd, hit_nk = [], [], []
    for batch in test_l:
        batch = tuple(t.to(DEVICE) for t in batch)
        uid, input_ids, answers = batch[0], batch[1], batch[2]
        _, kte, _, _, _, _ = teacher_quant(teacher, input_ids, answers)
        kap_te.append(kte.cpu().numpy())
        bidx = uid.cpu().numpy()
        for m, acc in ((kd, hit_kd), (nk, hit_nk)):
            h = m.predict(input_ids, uid)[:, -1, :]
            _, hit = ranks_hr(h @ m.item_embeddings.weight.T, answers, bidx, trm)
            acc.append(hit)
    kap_te = np.concatenate(kap_te)
    hit_kd = np.concatenate(hit_kd); hit_nk = np.concatenate(hit_nk)
    lo, mid, hi = terciles(kap_te)
    print(f"  {'tercile':8s} {'KD HR@10':>9} {'noKD':>8} {'gain':>8}  (kappa range)")
    for nm, msk in (("lo", lo), ("mid", mid), ("hi", hi), ("ALL", np.ones_like(lo))):
        gain = hit_kd[msk].mean() - hit_nk[msk].mean()
        print(f"  {nm:8s} {hit_kd[msk].mean():9.4f} {hit_nk[msk].mean():8.4f} {gain:+8.4f}  "
              f"[{kap_te[msk].min():.2f},{kap_te[msk].max():.2f}]")
    np.savetxt(f"results/kappa_d1_{ds}.csv",
               np.stack([kap_te, hit_kd, hit_nk], 1), delimiter=",",
               header="kappa,hit_kd,hit_nokd", comments="")
    return d3_pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ML-1M", "Beauty", "LastFM"])
    a = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("D0: eval seen-mask = trainers.py iteration(): "
          "rating_pred[train_matrix[batch_user_index].toarray()>0] = 0")
    print("    repeat stats (reported in debias S0): P(y==ell)=0, P(y in input)=0, all DS/splits.")
    verdicts = {}
    for ds in a.datasets:
        verdicts[ds] = run_ds(ds)
    print(f"\nD3 verdicts: {verdicts}")


if __name__ == "__main__":
    main()
