"""Sanity gates S0/S1 for the gated relative de-bias loss (kd_mode=debias).

S0: repeat statistics per dataset — P(y==ell), P(y in input) — and the gate w
    (mean / frac>0.5 / frac<0.2 / 10-bin hist) computed with the BASELINE-end
    embeddings (cmi_linear_<ds>_b0.pt = pure-PL student). Train and test splits.
S1: gradient-level equivalence on ML-1M (single process, model.eval, fixed
    batch — the only reproducibility mamba_ssm permits): every zero-hyper arm
    (none / margin lam_db=0 / bpr lam_db=0 / logit_margin m=0 / reweight g=0)
    must produce IDENTICAL student gradients to the pure-PL reference; every
    NONZERO arm must (a) produce different gradients, (b) have L_db > 0
    (margin/bpr), (c) w in [0,1].

Run from seqKD/src/:  python sanity_debias.py
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

CKPT = "../../BSARec/src/output"
DS = {
    "Beauty": dict(ckpt="BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt", heads="2", alpha="0.7", c="5",
                   drop="0.5", lam_pl="2.0", rank_k="50"),
    "LastFM": dict(ckpt="BSARec_LastFM_grid_lr0.001_a0.9_c3_h1.pt",  heads="1", alpha="0.9", c="3",
                   drop="0.5", lam_pl="1.0", rank_k="10"),
    "ML-1M":  dict(ckpt="BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt",  heads="1", alpha="0.3", c="9",
                   drop="0.2", lam_pl="0.5", rank_k="50"),
}


def build(ds, extra=()):
    from utils import parse_args, make_teacher_args, set_seed
    from dataset import get_seq_dic, get_dataloder, get_rating_matrix
    from model import MODEL_DICT
    cfg = DS[ds]
    sys.argv = ["sanity_debias.py", "--model_type", "kdstudent_v3", "--data_name", ds,
                "--train_name", f"sanity_debias_{ds}",
                "--do_distill", "--kd_mode", "debias",
                "--teacher_type", "bsarec", "--teacher_ckpt", os.path.join(CKPT, cfg["ckpt"]),
                "--teacher_num_attention_heads", cfg["heads"],
                "--teacher_alpha", cfg["alpha"], "--teacher_c", cfg["c"],
                "--alpha", cfg["alpha"], "--c", cfg["c"],
                "--d_state", "16", "--d_conv", "4", "--expand", "1",
                "--hidden_size", "64", "--num_hidden_layers", "2",
                "--hidden_dropout_prob", cfg["drop"], "--attention_probs_dropout_prob", cfg["drop"],
                "--lr", "0.001", "--batch_size", "256", "--seed", "42", "--gpu_id", "0",
                "--lambda_kd", cfg["lam_pl"], "--rank_k", cfg["rank_k"]] + list(extra)
    args = parse_args()
    set_seed(args.seed)
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1
    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name + '_same_target.npy')
    tr, ev, te = get_dataloder(args, seq_dic)
    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(ds, seq_dic, max_item)
    return args, seq_dic, tr, ev, te, MODEL_DICT, make_teacher_args


# ---------------------------------------------------------------- S0
@torch.no_grad()
def s0(device):
    print("=" * 72 + "\nS0: repeat statistics + baseline-end gate w distribution\n" + "=" * 72)
    print("eval seen-mask: trainers.py iteration() —")
    print("    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0")
    print("  -> seen items are removed from the ranking; with P(y==ell)=0 below,")
    print("     'ell can never be the correct test answer' holds.\n")
    for ds in DS:
        args, seq_dic, tr, ev, te, MD, _ = build(ds)
        b0 = os.path.join("output", f"cmi_linear_{ds}_b0.pt")
        state = torch.load(b0, map_location=device)
        E = state["item_embeddings.weight"].to(device)
        for split, loader in (("train", tr), ("test", te)):
            n = yl = yin = 0
            ws = []
            npad = 0
            for batch in loader:
                _, input_ids, answers = batch[0], batch[1].to(device), batch[2].to(device)
                ell = input_ids[:, -1]
                valid = ell != 0          # length-1 prefixes: all-padding input
                n += len(ell)
                npad += int((~valid).sum())
                yl += int((answers == ell)[valid].sum())
                yin += int((input_ids == answers[:, None]).any(1).sum())
                e_l = F.normalize(E[ell[valid]], dim=-1)
                e_y = F.normalize(E[answers[valid]], dim=-1)
                ws.append((1.0 - (e_l * e_y).sum(-1)).clamp(0, 1).cpu().numpy())
            w = np.concatenate(ws)
            hist = np.histogram(w, bins=10, range=(0, 1))[0] / len(w)
            print(f"{ds:7s} {split:5s} N={n:6d} (pad-last={100*npad/n:.1f}%)  "
                  f"P(y==ell)={yl/(n-npad):.4f}  P(y in input)={yin/n:.4f}  "
                  f"w: mean={w.mean():.3f} >0.5={100*(w>0.5).mean():.1f}% <0.2={100*(w<0.2).mean():.1f}%")
            print(f"         w hist(10bin): {np.round(hist, 3).tolist()}")
        print()


# ---------------------------------------------------------------- S1
def s1(device):
    from trainers import DebiasDistillTrainer
    from utils import set_logger
    print("=" * 72 + "\nS1: gradient-level equivalence (ML-1M, eval mode, fixed batch)\n" + "=" * 72)
    args, seq_dic, tr, ev, te, MD, make_teacher_args = build("ML-1M")
    logger = set_logger("output/sanity_debias.log")
    student = MD["kdstudent_v3"](args=args)
    ta = make_teacher_args(args)
    teacher = MD["bsarec"](args=ta)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    trainer = DebiasDistillTrainer(student, teacher, tr, ev, te, args, logger)
    batch = tuple(t.to(trainer.device) for t in next(iter(tr)))
    trainer.model.eval()           # determinism within process (G1-style)
    params = [p for p in trainer.model.parameters() if p.requires_grad]

    def grads(mode, active=True, **hp):
        trainer.debias_mode = mode
        trainer.lambda_db = hp.get("lambda_db", 1.0)
        trainer.margin_m = hp.get("margin_m", 0.3)
        trainer.lm_margin = hp.get("lm_margin", 1.0)
        trainer.gamma_rw = hp.get("gamma_rw", 1.0)
        trainer.detach_neg = hp.get("detach_neg", False)
        trainer._cur_epoch = trainer.db_warmup if active else 0
        trainer._ep = {}
        loss = trainer._compute_train_loss(batch)
        g = torch.autograd.grad(loss, params, allow_unused=True)
        flat = torch.cat([x.reshape(-1) for x in g if x is not None])
        ldb = trainer._ep.get("l_db", (float("nan"), 1))[0]
        w = trainer._gate(batch[1][:, -1], batch[2])
        return float(loss), flat, ldb, w

    _, gref, _, _ = grads("none")
    ok = True
    print("\n-- zero-hyper arms vs pure-PL reference (must be identical) --")
    for name, kw in [("margin lam_db=0", dict(lambda_db=0.0)),
                     ("bpr lam_db=0", dict(lambda_db=0.0)),
                     ("logit_margin m=0", dict(lm_margin=0.0)),
                     ("reweight g=0", dict(gamma_rw=0.0)),
                     ("none@warmup(inactive)", dict())]:
        mode = name.split()[0].replace("@warmup(inactive)", "none")
        active = "inactive" not in name
        _, g, _, _ = grads(mode if mode != "none" else "none", active=active, **kw)
        d = float((g - gref).abs().max())
        flag = "OK" if d < 1e-6 else "FAIL"
        ok &= d < 1e-6
        print(f"  {name:26s} max|grad diff| = {d:.3e}  [{flag}]")

    print("\n-- nonzero arms (must differ; L_db>0 for margin/bpr; w in [0,1]) --")
    for name, mode, kw in [("margin m=.3 lam=1", "margin", dict()),
                           ("margin detach_neg", "margin", dict(detach_neg=True)),
                           ("bpr lam=1", "bpr", dict()),
                           ("logit_margin m=1", "logit_margin", dict()),
                           ("reweight g=1", "reweight", dict())]:
        loss, g, ldb, w = grads(mode, **kw)
        d = float((g - gref).abs().max())
        wok = bool((w >= 0).all() and (w <= 1).all())
        ldb_s = f"L_db={ldb:.4f}" if mode in ("margin", "bpr") else "L_db=  n/a "
        cond = d > 1e-9 and wok and (mode not in ("margin", "bpr") or ldb > 0)
        ok &= cond
        print(f"  {name:26s} loss={loss:.4f} {ldb_s} max|grad diff|={d:.3e} "
              f"w_range=[{float(w.min()):.3f},{float(w.max()):.3f}]  [{'OK' if cond else 'FAIL'}]")
    print(f"\nS1 verdict: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s0(device)
    s1(device)
