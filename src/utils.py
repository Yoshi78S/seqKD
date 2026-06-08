import os
import random
import torch
import datetime
import argparse
import numpy as np
import logging


def set_logger(log_path, log_name='seqkd', mode='a'):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers across runs in the same process
    logger.handlers = []

    fh = logging.FileHandler(log_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')


def get_local_time():
    cur = datetime.datetime.now()
    return cur.strftime('%b-%d-%Y_%H-%M-%S')


def parse_args():
    parser = argparse.ArgumentParser()

    # basic args
    parser.add_argument("--data_dir", default="../../BSARec/src/data/", type=str,
                        help="defaults to BSARec data folder so we don't duplicate datasets")
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default=None, type=str)
    parser.add_argument("--train_name", default=get_local_time(), type=str)
    parser.add_argument("--num_items", default=10, type=int)
    parser.add_argument("--num_users", default=10, type=int)

    # train args
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--variance", default=5, type=float)

    # model args (shared)
    parser.add_argument("--model_type", default='mlp_student', type=str,
                        help="mlp_student | sigma | bsarec | duorec | gru4rec | lrurec | fmlprec")
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--num_hidden_layers", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)

    # SIGMA-specific (used when model_type == sigma OR teacher_type == sigma)
    parser.add_argument("--d_state", default=32, type=int)
    parser.add_argument("--d_conv", default=4, type=int)
    parser.add_argument("--expand", default=2, type=int)

    # BSARec-specific
    parser.add_argument("--alpha", default=0.9, type=float)
    parser.add_argument("--c", default=3, type=int)

    # DuoRec-specific
    parser.add_argument("--tau", default=1.0, type=float)
    parser.add_argument("--lmd", default=0.1, type=float)
    parser.add_argument("--lmd_sem", default=0.1, type=float)
    parser.add_argument("--ssl", default="us_x", type=str)
    parser.add_argument("--sim", default="dot", type=str)

    # GRU4Rec-specific
    parser.add_argument("--gru_hidden_size", default=64, type=int)

    # ===== Knowledge distillation args =====
    parser.add_argument("--do_distill", action="store_true",
                        help="Enable distillation (student is --model_type, teacher is --teacher_type).")
    parser.add_argument("--teacher_type", default="sigma", type=str)
    parser.add_argument("--teacher_ckpt", default=None, type=str,
                        help="Path to teacher .pt checkpoint. Required when --do_distill.")
    parser.add_argument("--lambda_kd", default=1.0, type=float)
    parser.add_argument("--kd_temperature", default=2.0, type=float)

    # Adaptive ranking distillation (kd_mode=adaptive_rank): L = L_rec + lambda_kd * L_pred^adapt.
    # No HS-KD. Teacher attention is temperature-manipulated into an order target
    # (sharpened) and a set target (flattened); a per-sample order-dependence rho
    # (pre-residual last-position attention entropy) interpolates them at the
    # score level; the top-K ranking is distilled via a Plackett-Luce loss
    # (rank-only -> invariant to student hidden dim).
    parser.add_argument("--rank_beta", default=1.0, type=float,
                        help="rank_naive: position-weight shape w=softmax(-rank/beta). "
                             "Large beta -> ~uniform (most naive); small -> top-concentrated.")
    parser.add_argument("--kd_mode", default="kl",
                        choices=["kl", "adaptive_rank", "adaptive_rank_v2",
                                 "adaptive_rank_comp", "rank_naive"],
                        help="kl = current KL pred-KD (+ optional HS-KD); "
                             "adaptive_rank = v1 (attention-temperature targets, deprecated); "
                             "adaptive_rank_v2 = pre/post-residual interpolation (diagnosed inert); "
                             "adaptive_rank_comp = complementary distillation "
                             "(main PL on z_ord + rho-gated complement from z_set); "
                             "rank_naive = RD-style pointwise ranking distillation.")
    # Complementary distillation (kd_mode=adaptive_rank_comp).
    parser.add_argument("--comp_beta", default=0.5, type=float,
                        help="weight of the complement term.")
    parser.add_argument("--comp_k", default=10, type=int,
                        help="top-K for the complement set (z_set rescues that z_ord drops).")
    parser.add_argument("--comp_use_hinge", action="store_true",
                        help="complement loss: hinge relu(margin - s) instead of -logsigmoid(s).")
    parser.add_argument("--comp_margin", default=0.0, type=float,
                        help="hinge margin (only when --comp_use_hinge).")
    parser.add_argument("--gate_fixed", default=-1.0, type=float,
                        help="ablation: -1 = adaptive gate (1+cos)/2; else gate = 1 - gate_fixed "
                             "(gate_fixed=1.0 -> gate 0 = complement off = pure ranking KD; "
                             "gate_fixed=0.0 -> gate 1 = uniform complement).")
    parser.add_argument("--tau_ord", default=0.5, type=float,
                        help="adaptive_rank (v1 only): attention temperature for the "
                             "order/sharpened teacher target (<1 sharpens attention).")
    parser.add_argument("--tau_set", default=2.0, type=float,
                        help="adaptive_rank (v1 only): attention temperature for the "
                             "set/flattened teacher target (>1 flattens attention).")
    parser.add_argument("--rank_k", default=50, type=int,
                        help="top-K for the Plackett-Luce ranking loss (v1 & v2).")
    parser.add_argument("--rho_measure", default="cos",
                        choices=["entropy", "cos", "jsd"],
                        help="order-dependence measure. v1: entropy (pre-residual attn "
                             "entropy). v2: cos (1 - cos(h_pre, h_post) at last position) "
                             "or jsd (JSD(softmax(z_ord), softmax(z_set))).")
    parser.add_argument("--teacher_pre_path", default="route1",
                        choices=["route1", "route2"],
                        help="adaptive_rank_v2: how z_set is built from the pre-residual "
                             "representation. route1 = final-block residual+LN skipped, "
                             "flowed through FFN->readout (z_ord = normal teacher). "
                             "route2 = h_pre / h_post dotted directly with item_emb "
                             "(symmetric, fallback if route1's z_set degrades).")
    parser.add_argument("--rho_fixed", default=-1.0, type=float,
                        help="adaptive_rank_v2 ablation: -1 = adaptive rho (default); "
                             "1.0 = z_ord only (post-residual); 0.0 = z_set only "
                             "(pre-residual). For the (b)/(c) comparison conditions.")

    # Hidden-state KD args
    parser.add_argument("--do_hs_distill", action="store_true",
                        help="Enable hidden-state KD (requires BSARec teacher + GRU4Rec student).")
    parser.add_argument("--lambda_hs", default=1.0, type=float)
    parser.add_argument("--hs_loss_type", default="mse", type=str,
                        choices=["mse", "cosine"])
    parser.add_argument("--hs_position_mode", default="all", type=str,
                        choices=["all", "last"])
    parser.add_argument("--hs_layer_mode", default="last", type=str,
                        choices=["last", "all"],
                        help="KDStudent HS-KD: 'last' aligns only the final "
                             "block's hidden state (default), 'all' averages "
                             "L_hs over every block.")
    parser.add_argument("--hs_use_projection", action="store_true")

    # KDStudent architecture ablation flags. Each disables one component.
    parser.add_argument("--abl_no_pos_emb", action="store_true",
                        help="KDStudent: drop position embeddings.")
    parser.add_argument("--abl_no_input_ln", action="store_true",
                        help="KDStudent: drop the LayerNorm + dropout right "
                             "after item_emb + pos_emb.")
    parser.add_argument("--abl_no_ffn", action="store_true",
                        help="KDStudent: drop the per-block FFN.")
    parser.add_argument("--abl_no_block_ln", action="store_true",
                        help="KDStudent: drop the LayerNorm after the GRU "
                             "inside each StudentBlock.")
    parser.add_argument("--abl_flat_gru", action="store_true",
                        help="KDStudent: replace StudentBlock × N with a "
                             "single nn.GRU(num_layers=N). Implies no FFN, no "
                             "block-internal LN, no HS-KD compatibility.")

    # KDStudent v2 ablation flags (only meaningful with model_type=kdstudent_v2)
    parser.add_argument("--abl_no_conv", action="store_true",
                        help="v2 only: drop the Linear→CausalConv1D in front "
                             "of the GRU.")
    parser.add_argument("--abl_no_gate", action="store_true",
                        help="v2 only: drop the SelectiveGate (uses raw GRU "
                             "output directly).")
    parser.add_argument("--abl_no_gated_mlp", action="store_true",
                        help="v2 only: drop the GatedMLP (uses identity in "
                             "its place, still with residual + LN).")

    # KDStudent v3 (FreqMamba) ablation flags (model_type=kdstudent_v3)
    parser.add_argument("--abl_no_freq", action="store_true",
                        help="v3 only: drop the frequency branch (pure Mamba; "
                             "alpha is ignored).")
    parser.add_argument("--abl_mamba_residual", action="store_true",
                        help="v3 only: re-add the fixed `+ x` residual on the "
                             "Mamba branch (tests the residual-dominance claim).")

    # Context-Direction Decorrelation (CDD) loss args
    parser.add_argument("--lambda_cdd", default=0.0, type=float,
                        help="Weight for the CDD loss. 0 disables it. CDD is "
                             "only computed in KDStudentDistillTrainer (i.e., "
                             "model_type=kdstudent or kdstudent_v2 with "
                             "--do_hs_distill).")
    parser.add_argument("--cdd_alpha", default=0.5, type=float,
                        help="Balance between L_align (context direction) and "
                             "L_uniform (representation spread) in CDD: "
                             "L_CDD = alpha * L_align + (1 - alpha) * L_uniform.")

    # Teacher architecture overrides — used only to build the teacher.
    # If left as -1 / None we copy the student's value.
    parser.add_argument("--teacher_hidden_size", default=-1, type=int)
    parser.add_argument("--teacher_num_hidden_layers", default=-1, type=int)
    parser.add_argument("--teacher_hidden_dropout_prob", default=-1.0, type=float)
    parser.add_argument("--teacher_attention_probs_dropout_prob", default=-1.0, type=float)
    parser.add_argument("--teacher_d_state", default=-1, type=int)
    parser.add_argument("--teacher_d_conv", default=-1, type=int)
    parser.add_argument("--teacher_expand", default=-1, type=int)
    parser.add_argument("--teacher_alpha", default=-1.0, type=float)
    parser.add_argument("--teacher_c", default=-1, type=int)
    parser.add_argument("--teacher_num_attention_heads", default=-1, type=int)
    parser.add_argument("--teacher_tau", default=-1.0, type=float)
    parser.add_argument("--teacher_lmd", default=-1.0, type=float)
    parser.add_argument("--teacher_lmd_sem", default=-1.0, type=float)
    parser.add_argument("--teacher_ssl", default=None, type=str)
    parser.add_argument("--teacher_sim", default=None, type=str)

    return parser.parse_args()


def make_teacher_args(args):
    """Return a shallow copy of `args` with teacher_* overrides applied.

    Architecture-shaping fields (hidden_size, num_hidden_layers, dropout, d_state,
    d_conv, expand) are taken from --teacher_* when provided; otherwise inherited
    from the student. Data-shaping fields (item_size, max_seq_length, batch_size,
    initializer_range) are always shared because they describe the dataset, not
    the model.
    """
    import copy
    targs = copy.copy(args)

    overrides = {
        "hidden_size": args.teacher_hidden_size,
        "num_hidden_layers": args.teacher_num_hidden_layers,
        "hidden_dropout_prob": args.teacher_hidden_dropout_prob,
        "attention_probs_dropout_prob": args.teacher_attention_probs_dropout_prob,
        "d_state": args.teacher_d_state,
        "d_conv": args.teacher_d_conv,
        "expand": args.teacher_expand,
        "alpha": args.teacher_alpha,
        "c": args.teacher_c,
        "num_attention_heads": args.teacher_num_attention_heads,
        "tau": args.teacher_tau,
        "lmd": args.teacher_lmd,
        "lmd_sem": args.teacher_lmd_sem,
        "ssl": args.teacher_ssl,
        "sim": args.teacher_sim,
    }
    for k, v in overrides.items():
        if v is None:
            continue
        if isinstance(v, (int, float)) and v < 0:
            continue
        setattr(targs, k, v)
    return targs


class EarlyStopping:
    def __init__(self, checkpoint_path, logger, patience=10, verbose=False, delta=0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.delta = delta
        self.logger = logger

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model, epoch=None):
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.best_epoch = epoch
            self.save_checkpoint(score, model, epoch)
        elif self.compare(score):
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model, epoch)
            self.counter = 0

    def save_checkpoint(self, score, model, epoch=None):
        if self.verbose:
            tag = f' (epoch {epoch})' if epoch is not None else ''
            self.logger.info(f'Validation score increased{tag}.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score
