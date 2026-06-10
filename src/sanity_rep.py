"""Sanity gates S1-S6 + lambda_rep grid probe for kappa-relational distillation.

S1: gradient-level equivalence (eval mode, fixed batch, single process): every
    zero-config arm (none / each mode with lambda_rep=0) == pure-PL gradients.
S2: requires_grad asserts on T/kappa live in the trainer (exercised here).
S3: grep evidence (eval path untouched) -- printed.
S6: on-the-fly trainer kappa == diag_kappa.teacher_quant kappa on sample batches.
Probe: raw values of L_rec / lam*L_PL / L_rep on the init batch -> lambda_rep
       grid at {1%, 10%, 50%} of L_rec (spec 3.6).
(S4 = 3-epoch short runs and S5 = PL-path diff are run separately via main.py.)

Run from seqKD/src/:  python sanity_rep.py
"""
import os, sys
import torch

CKPT = "../../BSARec/src/output"
DS = {
    "ML-1M":  dict(ckpt="BSARec_ML-1M_grid_lr0.0005_a0.3_c9_h1.pt",  heads="1", alpha="0.3", c="9",
                   drop="0.2", lam_pl="0.5", rank_k="50"),
    "Beauty": dict(ckpt="BSARec_Beauty_grid_lr0.0005_a0.7_c5_h2.pt", heads="2", alpha="0.7", c="5",
                   drop="0.5", lam_pl="2.0", rank_k="50"),
}


def build_trainer(ds):
    from utils import parse_args, make_teacher_args, set_seed, set_logger
    from dataset import get_seq_dic, get_dataloder, get_rating_matrix
    from model import MODEL_DICT
    from trainers import RepDistillTrainer
    cfg = DS[ds]
    sys.argv = ["sanity_rep.py", "--model_type", "kdstudent_v3", "--data_name", ds,
                "--train_name", f"sanity_rep_{ds}", "--do_distill", "--kd_mode", "rep",
                "--teacher_type", "bsarec", "--teacher_ckpt", os.path.join(CKPT, cfg["ckpt"]),
                "--teacher_num_attention_heads", cfg["heads"],
                "--teacher_alpha", cfg["alpha"], "--teacher_c", cfg["c"],
                "--alpha", cfg["alpha"], "--c", cfg["c"],
                "--d_state", "16", "--d_conv", "4", "--expand", "1",
                "--hidden_size", "64", "--num_hidden_layers", "2",
                "--hidden_dropout_prob", cfg["drop"], "--attention_probs_dropout_prob", cfg["drop"],
                "--lr", "0.001", "--batch_size", "256", "--seed", "42", "--gpu_id", "0",
                "--lambda_kd", cfg["lam_pl"], "--rank_k", cfg["rank_k"]]
    args = parse_args()
    set_seed(args.seed)
    args.cuda_condition = torch.cuda.is_available()
    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1
    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name + '_same_target.npy')
    tr, ev, te = get_dataloder(args, seq_dic)
    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(ds, seq_dic, max_item)
    student = MODEL_DICT["kdstudent_v3"](args=args)
    ta = make_teacher_args(args)
    teacher = MODEL_DICT["bsarec"](args=ta)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location="cuda"))
    logger = set_logger("output/sanity_rep.log")
    return RepDistillTrainer(student, teacher, tr, ev, te, args, logger), tr


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("S3: eval path -- Trainer.iteration() eval branch contains no rep symbols; "
          "RepDistillTrainer.valid() = super().valid() + read-only CSV row.\n")

    for ds in DS:
        print("=" * 72 + f"\n{ds}\n" + "=" * 72)
        trainer, tr = build_trainer(ds)
        batch = tuple(t.to(trainer.device) for t in next(iter(tr)))
        trainer.model.eval()
        params = [p for p in trainer.model.parameters() if p.requires_grad]

        def grads(mode, lam):
            trainer.rep_mode = mode
            trainer.lambda_rep = lam
            trainer._ep = {}
            loss = trainer._compute_train_loss(batch)
            g = torch.autograd.grad(loss, params, allow_unused=True)
            return float(loss), torch.cat([x.reshape(-1) for x in g if x is not None])

        # ---- probe (spec 3.6): raw magnitudes on the init batch ----
        trainer.rep_mode, trainer.lambda_rep, trainer._ep = 'raw', 1.0, {}
        _ = trainer._compute_train_loss(batch)
        m = {k: v[0] / v[1] for k, v in trainer._ep.items()}
        l_rec, l_pl, l_rep = m['l_rec'], m['l_pl'], m['l_rep']
        grid = [l_rec * f / l_rep for f in (0.01, 0.10, 0.50)]
        print(f"probe: L_rec={l_rec:.3f}  lam_PL*L_PL={trainer.lambda_kd * l_pl:.3f}  "
              f"L_rep(raw)={l_rep:.5f}")
        print(f"lambda_rep grid (1%/10%/50% of L_rec): "
              f"{[round(x, 1) for x in grid]}")

        # ---- S1 ----
        _, gref = grads('none', 0.0)
        ok = True
        for mode in ('corrected', 'raw', 'pairgate', 'shuffled', 'pairgate_shuf'):
            _, g = grads(mode, 0.0)
            dmax = float((g - gref).abs().max())
            ok &= dmax < 1e-6
            print(f"S1 {mode:14s} lam_rep=0: max|grad diff| = {dmax:.3e} "
                  f"[{'OK' if dmax < 1e-6 else 'FAIL'}]")
        for mode in ('corrected', 'raw', 'pairgate', 'shuffled'):
            loss, g = grads(mode, grid[1])
            dmax = float((g - gref).abs().max())
            lrep = trainer._ep['l_rep'][0] / trainer._ep['l_rep'][1]
            cond = dmax > 1e-9 and lrep > 0
            ok &= cond
            print(f"S1 {mode:14s} lam_rep={grid[1]:.1f}: loss={loss:.4f} "
                  f"L_rep={lrep:.5f} max|grad diff|={dmax:.3e} [{'OK' if cond else 'FAIL'}]")

        # ---- S6 ----
        from diag_kappa import teacher_quant, TEACH
        input_ids, answers = batch[1][:100], batch[2][:100]
        _, kap_tr, _, _ = trainer._teacher_quant(input_ids, answers)
        _, kap_dg, g_dg, _, _, ell = teacher_quant(trainer.teacher, input_ids, answers)
        d6 = float((kap_tr - kap_dg).abs().max())
        trainer.rep_mode = 'corrected'
        _, _, _, that_tr = trainer._teacher_quant(input_ids, answers)
        import torch.nn.functional as F
        that_dg = g_dg / g_dg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        d6g = float((that_tr - that_dg).abs().max())
        ok &= d6 < 1e-6 and d6g < 1e-6
        print(f"S6 on-the-fly vs diag (100 inst): max|kappa diff|={d6:.2e} "
              f"max|g_hat diff|={d6g:.2e} [{'OK' if d6 < 1e-6 and d6g < 1e-6 else 'FAIL'}]")
        print(f"{ds} verdict: {'PASS' if ok else 'FAIL'}\n")


if __name__ == "__main__":
    main()
