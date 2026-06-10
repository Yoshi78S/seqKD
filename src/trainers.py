import math
import os
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from metrics import recall_at_k, ndcg_k


class Trainer:
    """Single-model trainer (no distillation). Same evaluation protocol as BSARec."""

    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args, logger):
        super().__init__()

        self.args = args
        self.logger = logger
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr,
                          betas=betas, weight_decay=self.args.weight_decay)

        self.logger.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")

        self.train_epoch_times = []
        self.eval_epoch_times = []
        self.test_time = None

    def _sync(self):
        if self.cuda_condition:
            torch.cuda.synchronize()

    def train(self, epoch):
        self._sync()
        t0 = time.perf_counter()
        self.iteration(epoch, self.train_dataloader, train=True)
        self._sync()
        elapsed = time.perf_counter() - t0
        self.train_epoch_times.append(elapsed)
        self.logger.info(f"TIMING train_epoch {epoch} {elapsed:.4f}s")

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix
        self._sync()
        t0 = time.perf_counter()
        result = self.iteration(epoch, self.eval_dataloader, train=False)
        self._sync()
        elapsed = time.perf_counter() - t0
        self.eval_epoch_times.append(elapsed)
        self.logger.info(f"TIMING valid_epoch {epoch} {elapsed:.4f}s")
        return result

    def test(self, epoch):
        self.args.train_matrix = self.args.test_rating_matrix
        self._sync()
        t0 = time.perf_counter()
        result = self.iteration(epoch, self.test_dataloader, train=False)
        self._sync()
        elapsed = time.perf_counter() - t0
        self.test_time = elapsed
        self.logger.info(f"TIMING test {elapsed:.4f}s")
        return result

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location=self.device))

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HR@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HR@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3]),
        }
        self.logger.info(post_fix)
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def _compute_train_loss(self, batch):
        """Override point for distillation."""
        user_ids, input_ids, answers, neg_answer, same_target = batch
        return self.model.calculate_loss(input_ids, answers, neg_answer, same_target, user_ids)

    def iteration(self, epoch, dataloader, train=True):
        str_code = "train" if train else "test"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Mode_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")

        if train:
            self.model.train()
            rec_loss = 0.0
            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                loss = self._compute_train_loss(batch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                self.logger.info(str(post_fix))

        else:
            self.model.eval()
            pred_list = None
            answer_list = None

            with torch.no_grad():
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, answers, _, _ = batch
                    recommend_output = self.model.predict(input_ids, user_ids)
                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()

                    try:
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    except Exception:
                        rating_pred = rating_pred[:, :-1]
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            return self.get_full_sort_score(epoch, answer_list, pred_list)


class HiddenStateExtractor:
    """Captures intermediate hidden states via forward hooks."""

    def __init__(self):
        self.captured = {}
        self._hooks = []

    def register(self, name, module, detach=True):
        def hook_fn(mod, inp, out, _name=name, _detach=detach):
            val = out[0] if isinstance(out, tuple) else out
            self.captured[_name] = val.detach() if _detach else val
        self._hooks.append(module.register_forward_hook(hook_fn))

    def clear(self):
        self.captured.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class HiddenStateKDLoss(nn.Module):
    def __init__(self, teacher_dim=64, student_dim=64,
                 loss_type='mse', position_mode='all', use_projection=False):
        super().__init__()
        self.loss_type = loss_type
        self.position_mode = position_mode
        self.use_projection = use_projection
        if use_projection:
            self.projector = nn.Linear(student_dim, teacher_dim)

    def forward(self, teacher_hs, student_hs, valid_mask=None):
        if self.use_projection:
            student_hs = self.projector(student_hs)

        if self.position_mode == 'last':
            lengths = valid_mask.sum(dim=1).long()
            batch_idx = torch.arange(teacher_hs.shape[0], device=teacher_hs.device)
            last_pos = lengths - 1
            t = teacher_hs[batch_idx, last_pos]
            s = student_hs[batch_idx, last_pos]
            if self.loss_type == 'mse':
                return F.mse_loss(s, t)
            return (1 - F.cosine_similarity(s, t, dim=-1)).mean()

        # position_mode == 'all'
        if self.loss_type == 'mse':
            loss = F.mse_loss(student_hs, teacher_hs, reduction='none').mean(dim=-1)
        else:
            loss = 1 - F.cosine_similarity(student_hs, teacher_hs, dim=-1)

        if valid_mask is not None:
            return (loss * valid_mask).sum() / (valid_mask.sum() + 1e-10)
        return loss.mean()


class ContextDirectionDecorrelationLoss(nn.Module):
    """Context-Direction Decorrelation (CDD) loss.

    For each sample, take the last-position teacher representations and form
        delta_i = h^{T,pre}_i - h^{T,post}_i
    the direction of the *context information that the residual attenuated*.

    L_align   = -E[(δ̂_i · ŝ_i)^2]
        — pushes the student representation to lie along the δ axis (sign-
        invariant: +δ and -δ are both rewarded since the residual's sign is
        arbitrary).

    L_uniform = log E_{i≠j}[exp(-2||ŝ_i - ŝ_j||^2)]   (Wang & Isola, 2020)
        — spreads the student representations across the unit hypersphere.

    L_CDD = alpha * L_align + (1 - alpha) * L_uniform.

    All vectors are L2-normalised before being used.
    Always computed at the last valid position of each sequence.
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    @staticmethod
    def _last_position(x, valid_mask):
        """Gather x[i, L_i - 1] for each i, where L_i = valid_mask[i].sum()."""
        lengths = valid_mask.sum(dim=1).long()
        batch_idx = torch.arange(x.shape[0], device=x.device)
        last_pos = (lengths - 1).clamp(min=0)
        return x[batch_idx, last_pos]

    def forward(self, teacher_pre, teacher_post, student_repr, valid_mask):
        # 1) Reduce to the last valid position. [B, d] each.
        t_pre = self._last_position(teacher_pre, valid_mask)
        t_post = self._last_position(teacher_post, valid_mask)
        s_repr = self._last_position(student_repr, valid_mask)

        # 2) Context direction δ = pre - post, normalised.
        delta = t_pre - t_post
        delta_hat = F.normalize(delta, dim=-1, eps=1e-12)
        s_hat = F.normalize(s_repr, dim=-1, eps=1e-12)

        # 3) L_align = -mean(cos^2(δ̂, ŝ))  (sign-invariant alignment).
        cos = (delta_hat * s_hat).sum(dim=-1)
        l_align = -(cos ** 2).mean()

        # 4) L_uniform on the student-side representations.
        B = s_hat.shape[0]
        if B < 2:
            l_uniform = torch.zeros((), device=s_hat.device)
        else:
            sq_pdist = torch.cdist(s_hat, s_hat, p=2).pow(2)  # [B, B]
            mask = ~torch.eye(B, dtype=torch.bool, device=s_hat.device)
            l_uniform = torch.log(
                torch.exp(-2.0 * sq_pdist[mask]).mean() + 1e-10
            )

        return self.alpha * l_align + (1.0 - self.alpha) * l_uniform


class DistillTrainer(Trainer):
    """Trainer with predict-level cross-entropy distillation (SimRec-style).

    Total loss:
        L = L_rec + lambda_kd * L_KD

    where L_rec is the student's own item-CE on ground-truth `answers`, and
    L_KD is KL(softmax(teacher_logits/T) || log_softmax(student_logits/T)) * T^2,
    computed over the full item vocabulary at the last sequence position.
    """

    def __init__(self, student_model, teacher_model,
                 train_dataloader, eval_dataloader, test_dataloader,
                 args, logger):
        super().__init__(student_model, train_dataloader, eval_dataloader,
                         test_dataloader, args, logger)
        self.teacher = teacher_model
        if self.cuda_condition:
            self.teacher.cuda()
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.lambda_kd = args.lambda_kd
        self.kd_temperature = args.kd_temperature

        n_t = sum(p.nelement() for p in self.teacher.parameters())
        self.logger.info(f"Teacher Parameters (frozen): {n_t}")
        self.logger.info(f"KD config: lambda_kd={self.lambda_kd}, T={self.kd_temperature}")

    def _teacher_logits(self, input_ids, user_ids=None):
        with torch.no_grad():
            seq_out = self.teacher.predict(input_ids, user_ids)
            seq_out = seq_out[:, -1, :]
            item_emb = self.teacher.item_embeddings.weight
            return torch.matmul(seq_out, item_emb.transpose(0, 1))

    def _student_logits(self, input_ids, user_ids=None):
        seq_out = self.model.predict(input_ids, user_ids)
        seq_out = seq_out[:, -1, :]
        item_emb = self.model.item_embeddings.weight
        return torch.matmul(seq_out, item_emb.transpose(0, 1))

    def _compute_train_loss(self, batch):
        user_ids, input_ids, answers, neg_answer, same_target = batch

        student_logits = self._student_logits(input_ids, user_ids)
        l_rec = F.cross_entropy(student_logits, answers)

        teacher_logits = self._teacher_logits(input_ids, user_ids)

        T = self.kd_temperature
        log_p_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(teacher_logits / T, dim=-1)
        l_kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

        return l_rec + self.lambda_kd * l_kd


class HiddenStateDistillTrainer(DistillTrainer):
    """Multi-level KD: Prediction-level + Hidden-state alignment.

    Captures BSARec teacher's pre-residual Attention output (rich context,
    mixing ratio 0.83-0.92) and aligns it with GRU4Rec student's raw GRU
    hidden states.

    Total loss: L_rec + lambda_kd * L_pred + lambda_hs * L_hs
    """

    def __init__(self, student_model, teacher_model,
                 train_dataloader, eval_dataloader, test_dataloader,
                 args, logger):
        super().__init__(student_model, teacher_model,
                         train_dataloader, eval_dataloader, test_dataloader,
                         args, logger)

        self.lambda_hs = args.lambda_hs

        # Teacher hook: pre-residual attention output from last block's dense layer
        self.teacher_extractor = HiddenStateExtractor()
        last_block = self.teacher.item_encoder.blocks[-1]
        self.teacher_extractor.register(
            'pre_residual', last_block.layer.attention_layer.dense, detach=True)

        # Student hook: raw GRU output (before dense projection)
        self.student_extractor = HiddenStateExtractor()
        self.student_extractor.register(
            'gru_raw', self.model.gru_layers, detach=False)

        teacher_hs_dim = args.hidden_size
        student_hs_dim = args.gru_hidden_size
        self.hs_loss_fn = HiddenStateKDLoss(
            teacher_dim=teacher_hs_dim,
            student_dim=student_hs_dim,
            loss_type=args.hs_loss_type,
            position_mode=args.hs_position_mode,
            use_projection=args.hs_use_projection,
        )
        if self.cuda_condition:
            self.hs_loss_fn.cuda()

        if args.hs_use_projection:
            params = list(self.model.parameters()) + list(self.hs_loss_fn.projector.parameters())
            betas = (self.args.adam_beta1, self.args.adam_beta2)
            self.optim = Adam(params, lr=self.args.lr,
                              betas=betas, weight_decay=self.args.weight_decay)

        logger.info(f"HS-KD config: lambda_hs={self.lambda_hs}, "
                     f"loss={args.hs_loss_type}, pos={args.hs_position_mode}, "
                     f"proj={args.hs_use_projection}")

    def _compute_train_loss(self, batch):
        user_ids, input_ids, answers, neg_answer, same_target = batch

        self.teacher_extractor.clear()
        self.student_extractor.clear()

        # Student forward (hook captures GRU raw output)
        student_logits = self._student_logits(input_ids, user_ids)
        l_rec = F.cross_entropy(student_logits, answers)

        # Teacher forward (hook captures pre-residual attention output)
        teacher_logits = self._teacher_logits(input_ids, user_ids)

        # Prediction-level KD
        T = self.kd_temperature
        log_p_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(teacher_logits / T, dim=-1)
        l_pred = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

        # Hidden-state KD
        teacher_hs = self.teacher_extractor.captured['pre_residual']
        student_hs = self.student_extractor.captured['gru_raw']
        valid_mask = (input_ids > 0).float()
        l_hs = self.hs_loss_fn(teacher_hs, student_hs, valid_mask=valid_mask)

        return l_rec + self.lambda_kd * l_pred + self.lambda_hs * l_hs


class KDStudentDistillTrainer(DistillTrainer):
    """Multi-level KD for KDStudent: per-block alignment with BSARec teacher.

    Teacher hooks: each block's attention_layer.dense (pre-residual)
    Student hooks: each block's GRU (pre-dropout/LN)
    """

    def __init__(self, student_model, teacher_model,
                 train_dataloader, eval_dataloader, test_dataloader,
                 args, logger):
        super().__init__(student_model, teacher_model,
                         train_dataloader, eval_dataloader, test_dataloader,
                         args, logger)

        self.lambda_hs = args.lambda_hs

        self.teacher_extractor = HiddenStateExtractor()
        for i, block in enumerate(self.teacher.item_encoder.blocks):
            # `layer_{i}` is the pre-residual attention output (W_O projection),
            # used by HS-KD. Kept under the existing name for backward compat.
            self.teacher_extractor.register(
                f'layer_{i}', block.layer.attention_layer.dense, detach=True)
            # `post_{i}` is the post-residual + LayerNorm output of the
            # attention sub-layer. Used by CDD as the "after-residual" anchor.
            self.teacher_extractor.register(
                f'post_{i}', block.layer.attention_layer, detach=True)

        # Each KDStudent block exposes `hs_hook_target` — the module whose
        # forward output is the block's hidden state for HS-KD / CDD alignment.
        # v1 student → block.gru, v2 student → block.selective_gate.
        self.student_extractor = HiddenStateExtractor()
        for i, block in enumerate(self.model.student_encoder.blocks):
            target = getattr(block, 'hs_hook_target', None)
            if target is None:
                raise AttributeError(
                    f"Student block #{i} has no `hs_hook_target` attribute; "
                    "cannot register HS-KD hook.")
            self.student_extractor.register(
                f'layer_{i}', target, detach=False)

        self.hs_loss_fn = HiddenStateKDLoss(
            teacher_dim=args.hidden_size,
            student_dim=args.hidden_size,
            loss_type=args.hs_loss_type,
            position_mode=args.hs_position_mode,
            use_projection=args.hs_use_projection,
        )
        if self.cuda_condition:
            self.hs_loss_fn.cuda()

        self.hs_layer_mode = args.hs_layer_mode

        # Context-Direction Decorrelation (CDD) — optional, computed on the
        # last block's representations.
        self.lambda_cdd = getattr(args, 'lambda_cdd', 0.0)
        self.cdd_alpha = getattr(args, 'cdd_alpha', 0.5)
        if self.lambda_cdd > 0:
            self.cdd_loss_fn = ContextDirectionDecorrelationLoss(
                alpha=self.cdd_alpha)
            if self.cuda_condition:
                self.cdd_loss_fn.cuda()
        else:
            self.cdd_loss_fn = None

        logger.info(f"KDStudent HS-KD config: lambda_hs={self.lambda_hs}, "
                     f"loss={args.hs_loss_type}, pos={args.hs_position_mode}, "
                     f"layer={self.hs_layer_mode}")
        if self.lambda_cdd > 0:
            logger.info(
                f"KDStudent CDD config: lambda_cdd={self.lambda_cdd}, "
                f"alpha={self.cdd_alpha}")

    def _compute_train_loss(self, batch):
        user_ids, input_ids, answers, neg_answer, same_target = batch

        self.teacher_extractor.clear()
        self.student_extractor.clear()

        student_logits = self._student_logits(input_ids, user_ids)
        l_rec = F.cross_entropy(student_logits, answers)

        teacher_logits = self._teacher_logits(input_ids, user_ids)

        T = self.kd_temperature
        log_p_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(teacher_logits / T, dim=-1)
        l_pred = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

        valid_mask = (input_ids > 0).float()
        num_blocks = len(self.model.student_encoder.blocks)

        if self.hs_layer_mode == 'all':
            l_hs_sum = 0
            for i in range(num_blocks):
                t = self.teacher_extractor.captured[f'layer_{i}']
                s = self.student_extractor.captured[f'layer_{i}']
                l_hs_sum = l_hs_sum + self.hs_loss_fn(t, s, valid_mask=valid_mask)
            l_hs = l_hs_sum / num_blocks
        else:
            last_idx = num_blocks - 1
            teacher_hs = self.teacher_extractor.captured[f'layer_{last_idx}']
            student_hs = self.student_extractor.captured[f'layer_{last_idx}']
            l_hs = self.hs_loss_fn(teacher_hs, student_hs, valid_mask=valid_mask)

        loss = l_rec + self.lambda_kd * l_pred + self.lambda_hs * l_hs

        # Context-Direction Decorrelation (CDD) — uses the last block's
        # pre/post and the student's last block representation.
        if self.cdd_loss_fn is not None:
            last_idx = num_blocks - 1
            t_pre = self.teacher_extractor.captured[f'layer_{last_idx}']
            t_post = self.teacher_extractor.captured[f'post_{last_idx}']
            s_repr = self.student_extractor.captured[f'layer_{last_idx}']
            l_cdd = self.cdd_loss_fn(t_pre, t_post, s_repr, valid_mask)
            loss = loss + self.lambda_cdd * l_cdd

        return loss


class AdaptiveRankingDistillTrainer(DistillTrainer):
    """Adaptive ranking distillation:  L = L_rec + lambda_kd * L_pred^adapt.

    No hidden-state KD, no CDD. The teacher (BSARec) attention distribution is
    temperature-manipulated to produce two prediction targets:
        z_ord = teacher with SHARPENED attention (tau_ord < 1)  — order-emphasized
        z_set = teacher with FLATTENED attention (tau_set > 1)  — set-emphasized
    A per-sample order-dependence rho in [0,1], measured from the teacher's
    PRE-residual last-position attention entropy (so it is independent of the
    residual-dominance bias), interpolates the two at the score level:
        z_adapt = rho * z_ord + (1 - rho) * z_set
    The interpolated top-K ranking is distilled to the student via a
    Plackett-Luce listwise loss. Only the ranking ORDER is transferred, so the
    loss is invariant to the student hidden dim (compression-friendly; no
    projection needed).

    Attention temperature is applied via a forward_pre_hook on each teacher
    block's `attention_layer.softmax` (divides the masked scores by tau). The
    teacher's two branches (alpha*frequency + (1-alpha)*attention) mean the
    temperature only moves the (1-alpha) attention branch — strongest signal on
    attention-heavy teachers (e.g. ML-1M, alpha=0.3), weaker on freq-heavy ones
    (LastFM alpha=0.9).
    """

    def __init__(self, student_model, teacher_model,
                 train_dataloader, eval_dataloader, test_dataloader,
                 args, logger):
        super().__init__(student_model, teacher_model,
                         train_dataloader, eval_dataloader, test_dataloader,
                         args, logger)
        self.tau_ord = args.tau_ord
        self.tau_set = args.tau_set
        self.rank_k = args.rank_k
        self.rho_measure = getattr(args, 'rho_measure', 'entropy')

        # Shared state read by the attention-softmax hooks.
        self._attn_tau = 1.0          # 1.0 => pre-hook is a no-op
        self._capture_probs = False
        self._attn_probs = {}         # layer_idx -> attention probs (baseline pass)
        self._attn_hooks = []
        for i, block in enumerate(self.teacher.item_encoder.blocks):
            softmax_mod = block.layer.attention_layer.softmax
            self._attn_hooks.append(
                softmax_mod.register_forward_pre_hook(self._tau_prehook))
            self._attn_hooks.append(
                softmax_mod.register_forward_hook(self._make_capture_hook(i)))

        logger.info(
            f"AdaptiveRanking config: lambda_pred={self.lambda_kd}, "
            f"tau_ord={self.tau_ord}, tau_set={self.tau_set}, "
            f"K={self.rank_k}, rho={self.rho_measure}")

    # ── attention-temperature + capture hooks on teacher softmax ──
    def _tau_prehook(self, _mod, inputs):
        if self._attn_tau == 1.0:
            return None  # no-op
        return (inputs[0] / self._attn_tau,) + tuple(inputs[1:])

    def _make_capture_hook(self, idx):
        def hook(_mod, _inp, out):
            if self._capture_probs:
                self._attn_probs[idx] = out.detach()
        return hook

    @torch.no_grad()
    def _teacher_logits_at_tau(self, input_ids, user_ids, tau, capture=False):
        """Full-vocab last-position teacher logits with attention temperature tau."""
        self._attn_tau = tau
        self._capture_probs = capture
        if capture:
            self._attn_probs = {}
        seq_out = self.teacher.predict(input_ids, user_ids)[:, -1, :]
        item_emb = self.teacher.item_embeddings.weight
        logits = torch.matmul(seq_out, item_emb.transpose(0, 1))
        self._attn_tau = 1.0
        self._capture_probs = False
        return logits

    def _compute_rho(self, input_ids):
        """Per-sample order-dependence rho in [0,1] from the baseline
        last-position attention entropy, averaged over heads and layers.

        rho = 1 - H(a_last) / log(L_valid).  rho->1 concentrated (order),
        rho->0 diffuse (set). Sequences of length 1 -> rho = 1.
        """
        valid_len = (input_ids > 0).sum(dim=1)                       # [B]
        log_len = torch.log(valid_len.float().clamp(min=2.0))        # avoid /0
        rhos = []
        for probs in self._attn_probs.values():
            a = probs[:, :, -1, :].clamp_min(1e-12)                  # [B, H, L]
            ent = -(a * a.log()).sum(dim=-1)                         # [B, H]
            rho_layer = 1.0 - ent / log_len[:, None]                 # [B, H]
            rhos.append(rho_layer.mean(dim=1))                       # [B]
        rho = torch.stack(rhos, dim=0).mean(dim=0).clamp(0.0, 1.0)   # [B]
        rho = torch.where(valid_len <= 1, torch.ones_like(rho), rho)
        return rho

    @staticmethod
    def _plackett_luce_loss(student_ranked):
        """student_ranked: [B, K] student scores in the teacher-ranked
        (descending) order. PL listwise NLL, per-position mean:
            L = (1/K) sum_k [ logsumexp(s_{k..K-1}) - s_k ]   (then mean over batch).
        The 1/K normalization (vs the raw sum in the spec) keeps the loss scale
        O(1) — comparable to L_rec — and decouples it from K, so lambda_pred is
        transferable across K. The constant 1/K is absorbed into lambda_pred and
        does not affect the rank-only / dim-invariance properties.
        """
        flipped = torch.flip(student_ranked, dims=[1])
        suffix_lse = torch.flip(torch.logcumsumexp(flipped, dim=1), dims=[1])
        per_sample = (suffix_lse - student_ranked).mean(dim=1)       # [B]
        return per_sample.mean()

    def _compute_train_loss(self, batch):
        user_ids, input_ids, answers, neg_answer, same_target = batch

        # Student: recommendation loss + full-vocab logits for ranking.
        student_logits = self._student_logits(input_ids, user_ids)   # [B, V]
        l_rec = F.cross_entropy(student_logits, answers)

        # Teacher: baseline pass (rho), then sharpened (ord) and flattened (set).
        _ = self._teacher_logits_at_tau(input_ids, user_ids, tau=1.0, capture=True)
        rho = self._compute_rho(input_ids)                           # [B]
        z_ord = self._teacher_logits_at_tau(input_ids, user_ids, tau=self.tau_ord)
        z_set = self._teacher_logits_at_tau(input_ids, user_ids, tau=self.tau_set)

        # Score-level adaptive interpolation -> target top-K ranking.
        z_adapt = rho[:, None] * z_ord + (1.0 - rho[:, None]) * z_set  # [B, V]
        k = min(self.rank_k, z_adapt.size(1))
        topk_idx = torch.topk(z_adapt, k, dim=1).indices              # [B, K]
        student_ranked = torch.gather(student_logits, 1, topk_idx)    # [B, K]
        l_pred = self._plackett_luce_loss(student_ranked)

        return l_rec + self.lambda_kd * l_pred


class AdaptiveRankingV2Trainer(DistillTrainer):
    """Adaptive ranking distillation v2:  L = L_rec + lambda_kd * L_pred^adapt.

    Corrects v1 (attention-temperature targets, which collapsed because the
    final-logit top-K is nearly invariant to attention temperature). v2 builds
    the two targets from the teacher's PRE-residual vs POST-residual attention
    representation — the exact point the mixing-ratio analysis showed a large
    structural gap (pre share ~0.93 vs post ~0.33-0.67), so z_ord != z_set is
    guaranteed:

        z_ord = teacher logits with the normal POST-residual attention
                (= the standard teacher; residual present, last-item dominant)
        z_set = teacher logits built from the PRE-residual attention output
                h_pre (context-rich, last-item dependence stripped)

    Per-sample order-dependence rho measures the pre/post divergence (large =
    residual dominates that sequence's prediction = order-dependent):
        rho_cos = 1 - cos(h_pre_last, h_post_last)          (--rho_measure cos)
        rho_jsd = JSD(softmax(z_ord), softmax(z_set)) / ln2 (--rho_measure jsd)
        z_adapt = rho * z_ord + (1 - rho) * z_set
    The top-K ranking of z_adapt is distilled via the Plackett-Luce listwise
    loss (rank-only -> dim-invariant; reuses AdaptiveRankingDistillTrainer).

    NO attention-temperature, NO residual-scaling (alpha untouched), NO HS-KD.
    h_pre/h_post are only READ via hooks; the fusion coefficient alpha and the
    residual are never modified.

    z_set construction (--teacher_pre_path):
      route1 (default): hook the FINAL block's attention_layer to return h_pre
        (skipping its out_dropout + LN(.+input)); the modified block output
        flows through the block FFN -> readout. z_ord = a normal forward.
      route2 (fallback): z_set = h_pre_last @ E^T, z_ord = h_post_last @ E^T
        (symmetric direct readout; use if route1's z_set degrades — diag B').
    """

    def __init__(self, student_model, teacher_model,
                 train_dataloader, eval_dataloader, test_dataloader,
                 args, logger):
        super().__init__(student_model, teacher_model,
                         train_dataloader, eval_dataloader, test_dataloader,
                         args, logger)
        self.rank_k = args.rank_k
        self.rho_measure = getattr(args, 'rho_measure', 'cos')
        self.pre_path = getattr(args, 'teacher_pre_path', 'route1')
        self.rho_fixed = getattr(args, 'rho_fixed', -1.0)

        # Final teacher block's attention sub-layer (pre-residual hook point).
        self._attn = self.teacher.item_encoder.blocks[-1].layer.attention_layer
        self._set_mode = False
        self._h_pre = None    # captured dense() output (pre-residual)   [B, L, d]
        self._h_post = None   # captured attention_layer output (post)    [B, L, d]
        self._attn.dense.register_forward_hook(self._cap_pre_hook)
        self._attn.register_forward_hook(self._attn_out_hook)

        logger.info(
            f"AdaptiveRankingV2 config: lambda_pred={self.lambda_kd}, "
            f"K={self.rank_k}, rho={self.rho_measure}, path={self.pre_path}, "
            f"rho_fixed={self.rho_fixed}")

    # ── hooks on the final block's attention sub-layer ──
    def _cap_pre_hook(self, _mod, _inp, out):
        # out = self.dense(context_layer): pre-residual attention output (h_pre)
        self._h_pre = out

    def _attn_out_hook(self, _mod, _inp, out):
        if self._set_mode:
            # Replace the attention sub-layer output (h_post = LN(h_pre+input))
            # with the pre-residual h_pre: skip residual + LN. Downstream
            # (alpha-fusion -> block FFN -> readout) then produces z_set.
            return self._h_pre
        self._h_post = out  # normal pass: capture h_post
        return None

    @torch.no_grad()
    def _teacher_logits(self, input_ids, user_ids=None, set_mode=False):
        self._set_mode = set_mode
        seq_out = self.teacher.predict(input_ids, user_ids)[:, -1, :]
        logits = torch.matmul(seq_out, self.teacher.item_embeddings.weight.transpose(0, 1))
        self._set_mode = False
        return logits

    def _compute_rho(self, h_pre_last, h_post_last, z_ord, z_set):
        if self.rho_fixed >= 0.0:
            return torch.full((h_pre_last.shape[0],), self.rho_fixed,
                              device=h_pre_last.device)
        if self.rho_measure == 'jsd':
            p = F.softmax(z_ord, dim=-1)
            q = F.softmax(z_set, dim=-1)
            m = 0.5 * (p + q)
            kl_pm = (p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
            kl_qm = (q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
            rho = (0.5 * (kl_pm + kl_qm) / math.log(2.0))
        else:  # cos
            cos = F.cosine_similarity(h_pre_last, h_post_last, dim=-1)
            rho = 1.0 - cos
        return rho.clamp(0.0, 1.0)

    def _compute_train_loss(self, batch):
        user_ids, input_ids, answers, neg_answer, same_target = batch

        student_logits = self._student_logits(input_ids, user_ids)   # [B, V]
        l_rec = F.cross_entropy(student_logits, answers)

        # Normal teacher pass: z_ord + capture final-block h_pre / h_post.
        z_ord = self._teacher_logits(input_ids, user_ids, set_mode=False)
        h_pre_last = self._h_pre[:, -1, :]
        h_post_last = self._h_post[:, -1, :]

        if self.pre_path == 'route2':
            E = self.teacher.item_embeddings.weight
            z_set = torch.matmul(h_pre_last, E.transpose(0, 1))
            z_ord = torch.matmul(h_post_last, E.transpose(0, 1))     # symmetric
        else:  # route1
            z_set = self._teacher_logits(input_ids, user_ids, set_mode=True)

        rho = self._compute_rho(h_pre_last, h_post_last, z_ord, z_set)  # [B]
        z_adapt = rho[:, None] * z_ord + (1.0 - rho[:, None]) * z_set

        k = min(self.rank_k, z_adapt.size(1))
        topk_idx = torch.topk(z_adapt, k, dim=1).indices
        student_ranked = torch.gather(student_logits, 1, topk_idx)
        l_pred = AdaptiveRankingDistillTrainer._plackett_luce_loss(student_ranked)

        return l_rec + self.lambda_kd * l_pred


class AdaptiveRankingCompTrainer(AdaptiveRankingV2Trainer):
    """Complementary distillation (NOT the interpolation of v2):

        L = L_rec + lambda_kd * [ L_rank(z^S, z_ord) + beta * g(s) * L_comp(z^S, z_set) ]

    Main term  L_rank : Plackett-Luce on the student's scores at z_ord's top-K
                        (z_ord = normal teacher, the strong predictor).
    Complement L_comp : pulls UP only the items that z_set's top-K caught but
                        z_ord's top-K dropped (z_set = pre-residual / residual-
                        skipped view, which the diagnostic showed rescues items
                        in set-dependent sequences). z_set is NOT imitated whole.
    Gate g(s)         : (1 + cos(h_pre, h_post)) / 2 in [0,1] — large for
                        set-dependent (high cos = low rho) sequences, where the
                        complementarity diagnostic localised the rescues.
                        --gate_fixed overrides: gate = 1 - gate_fixed
                        (1.0 -> gate 0 = complement off = pure ranking KD;
                         0.0 -> gate 1 = uniform complement).

    Reuses v2's pre/post hooks, _teacher_logits(set_mode) (route1/route2) and the
    Plackett-Luce loss. Rank-only -> dim-invariant. No residual scaling, no HS-KD.

    K note: the main ranking uses --rank_k (default 50, richer signal); the
    complement uses --comp_k (default 10, the recommendation-relevant top set the
    diagnostic measured). Pass --rank_k 10 to match the spec's single-K form.
    """

    def __init__(self, student_model, teacher_model,
                 train_dataloader, eval_dataloader, test_dataloader,
                 args, logger):
        super().__init__(student_model, teacher_model,
                         train_dataloader, eval_dataloader, test_dataloader,
                         args, logger)
        self.comp_beta = getattr(args, 'comp_beta', 0.5)
        self.comp_k = getattr(args, 'comp_k', 10)
        self.comp_use_hinge = getattr(args, 'comp_use_hinge', False)
        self.comp_margin = getattr(args, 'comp_margin', 0.0)
        self.gate_fixed = getattr(args, 'gate_fixed', -1.0)
        logger.info(
            f"AdaptiveRankingComp config: beta={self.comp_beta}, comp_k={self.comp_k}, "
            f"hinge={self.comp_use_hinge}, margin={self.comp_margin}, "
            f"gate_fixed={self.gate_fixed}, rank_k={self.rank_k}, path={self.pre_path}")

    def _compute_train_loss(self, batch):
        user_ids, input_ids, answers, neg_answer, same_target = batch

        student_logits = self._student_logits(input_ids, user_ids)        # [B, V]
        l_rec = F.cross_entropy(student_logits, answers)

        # Normal teacher pass: z_ord + capture final-block h_pre / h_post.
        z_ord = self._teacher_logits(input_ids, user_ids, set_mode=False)
        h_pre_last = self._h_pre[:, -1, :]
        h_post_last = self._h_post[:, -1, :]
        if self.pre_path == 'route2':
            E = self.teacher.item_embeddings.weight
            z_set = torch.matmul(h_pre_last, E.transpose(0, 1))
        else:  # route1
            z_set = self._teacher_logits(input_ids, user_ids, set_mode=True)

        # ── main term: PL ranking on z_ord's top-(rank_k) ──
        kr = min(self.rank_k, z_ord.size(1))
        topk_ord_rank = torch.topk(z_ord, kr, dim=1).indices
        student_ranked = torch.gather(student_logits, 1, topk_ord_rank)
        l_rank = AdaptiveRankingDistillTrainer._plackett_luce_loss(student_ranked)

        # ── gate g(s) ──
        if self.gate_fixed >= 0.0:
            gate = torch.full((student_logits.size(0),), 1.0 - self.gate_fixed,
                              device=student_logits.device)
        else:
            cos = F.cosine_similarity(h_pre_last, h_post_last, dim=-1)
            gate = ((1.0 + cos) / 2.0).clamp(0.0, 1.0)

        # ── complement term: lift items z_set caught but z_ord dropped ──
        kc = min(self.comp_k, z_ord.size(1))
        topk_ord = torch.topk(z_ord, kc, dim=1).indices       # [B, kc]
        topk_set = torch.topk(z_set, kc, dim=1).indices       # [B, kc]
        comp_terms = []
        for b in range(student_logits.size(0)):
            diff = topk_set[b][~torch.isin(topk_set[b], topk_ord[b])]
            if diff.numel() > 0:
                s = student_logits[b, diff]
                if self.comp_use_hinge:
                    comp_terms.append(F.relu(self.comp_margin - s).mean())
                else:
                    comp_terms.append(-F.logsigmoid(s).mean())
            else:
                comp_terms.append(student_logits.new_zeros(()))
        l_comp = (gate * torch.stack(comp_terms)).mean()

        l_pred = l_rank + self.comp_beta * l_comp
        return l_rec + self.lambda_kd * l_pred


class RankNaiveDistillTrainer(DistillTrainer):
    """RD-style naive POINTWISE ranking distillation (Tang & Wang, 2018).

        L = L_rec + lambda_kd * L_rank^naive
        L_rank^naive = - sum_{i in TopK(z_ord)} w_i * log sigmoid(z^S_i)
        w_i = softmax(-rank_i / beta)   (rank_i = 1..K; beta large -> ~uniform
                                         = most naive; beta small -> top-concentrated
                                         = RD original)

    The teacher's top-K are treated as extra positives whose student scores are
    pushed up; L_rec's softmax pushes everything else down (the RD tug-of-war).
    Pointwise (no list/pair structure) — distinct from the PL listwise loss.
    Uses only z_ord (normal teacher post logits); no pre/post hooks, no HS-KD.
    Rank-only on the student side via logsigmoid -> compression-friendly.
    """

    def __init__(self, student_model, teacher_model,
                 train_dataloader, eval_dataloader, test_dataloader,
                 args, logger):
        super().__init__(student_model, teacher_model,
                         train_dataloader, eval_dataloader, test_dataloader,
                         args, logger)
        self.rank_k = args.rank_k
        self.rank_beta = getattr(args, 'rank_beta', 1.0)
        logger.info(f"RankNaive config: lambda_rank={self.lambda_kd}, "
                    f"K={self.rank_k}, beta={self.rank_beta}")

    def _compute_train_loss(self, batch):
        user_ids, input_ids, answers, neg_answer, same_target = batch

        student_logits = self._student_logits(input_ids, user_ids)        # [B, V]
        l_rec = F.cross_entropy(student_logits, answers)

        z_ord = self._teacher_logits(input_ids, user_ids)                 # [B, V] post
        k = min(self.rank_k, z_ord.size(1))
        topk = torch.topk(z_ord, k, dim=-1).indices                       # [B, K]
        ranks = torch.arange(1, k + 1, device=z_ord.device, dtype=torch.float)
        w = torch.softmax(-ranks / self.rank_beta, dim=0)                 # [K]
        s = torch.gather(student_logits, 1, topk)                         # [B, K]
        l_rank = -(w.unsqueeze(0) * F.logsigmoid(s)).sum(dim=1).mean()

        return l_rec + self.lambda_kd * l_rank


# ── Conditional-IB corrective distillation (CEB term: beta * I(ell; h | y)) ──

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lambd, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class LastItemCritic(nn.Module):
    """q_theta(ell | h, y): predicts the last-item id from (h, E_y)."""
    def __init__(self, d, n_items, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_items),
        )

    def forward(self, h, e_y):
        return self.net(torch.cat([h, e_y], dim=-1))


class CMIDistillTrainer(DistillTrainer):
    """Conditional-IB corrective distillation.

        L = L_rec + lambda_kd * L_PL + cmi_beta * L_CEB,  L_CEB estimates I(ell; h | y)

    Base distillation is PL listwise ranking on the NORMAL teacher's top-K (z_ord)
    — the adopted method (NOT KL). The CEB term penalizes last-item (ell)
    information in the student readout h that is NOT explained by the target y
    (conditioning on y auto-gates: useful last-item info is kept). Teacher is used
    only for L_PL; the CEB term uses y only as the "noise definition". Critic and y
    are TRAIN-only — eval/predict never see them (G4).

    estimators (--cmi_estimator):
      none  : L_CEB = 0  (== pure PL base; the G1 reference)
      linear: penalize h's component along E_ell minus its E_y projection  (no critic)
      adv   : gradient-reversal critic q(ell|h,y); GRL strength = cmi_beta  (G1: beta=0
              -> zero student grad from CEB -> identical to PL). critic in student optim.
      club  : conditional CLUB upper bound on I(ell;h|y); critic by MLE (separate optim).
    """

    def __init__(self, student_model, teacher_model,
                 train_dataloader, eval_dataloader, test_dataloader, args, logger):
        super().__init__(student_model, teacher_model,
                         train_dataloader, eval_dataloader, test_dataloader, args, logger)
        self.cmi_estimator = getattr(args, 'cmi_estimator', 'none')
        self.cmi_beta = getattr(args, 'cmi_beta', 0.0)
        self.rank_k = getattr(args, 'rank_k', 50)
        self.critic = None
        self._leak_log = []
        self._samey_rate = float('nan')
        d, n_items = args.hidden_size, args.item_size
        if self.cmi_estimator in ('adv', 'club'):
            self.critic = LastItemCritic(d, n_items, getattr(args, 'cmi_hidden', 256))
            if self.cuda_condition:
                self.critic.cuda()
            betas = (self.args.adam_beta1, self.args.adam_beta2)
            if self.cmi_estimator == 'adv':
                # critic learns jointly with the student via GRL -> one optimizer
                self.optim = Adam(list(self.model.parameters()) + list(self.critic.parameters()),
                                  lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
            else:  # club: critic trained by MLE in a separate optimizer
                self.opt_critic = Adam(self.critic.parameters(),
                                       lr=getattr(args, 'cmi_critic_lr', 1e-3))
                self.cmi_critic_steps = getattr(args, 'cmi_critic_steps', 1)
        logger.info(f"CMI config: estimator={self.cmi_estimator}, beta={self.cmi_beta}, "
                    f"rank_k={self.rank_k}, base=PL")

    def _pl_term(self, student_logits, z_ord):
        k = min(self.rank_k, z_ord.size(1))
        topk = torch.topk(z_ord, k, dim=1).indices
        return AdaptiveRankingDistillTrainer._plackett_luce_loss(
            torch.gather(student_logits, 1, topk))

    def _compute_train_loss(self, batch):
        user_ids, input_ids, answers, neg_answer, same_target = batch
        ell = input_ids[:, -1]
        E = self.model.item_embeddings.weight
        h = self.model.predict(input_ids, user_ids)[:, -1, :]          # [B, d]
        student_logits = torch.matmul(h, E.transpose(0, 1))
        l_rec = F.cross_entropy(student_logits, answers)
        z_ord = self._teacher_logits(input_ids, user_ids)              # normal teacher
        loss = l_rec + self.lambda_kd * self._pl_term(student_logits, z_ord)

        est = self.cmi_estimator
        if est == 'linear':
            ey = F.normalize(E[answers].detach(), dim=-1)
            el = E[ell].detach()
            el_perp = el - (el * ey).sum(-1, keepdim=True) * ey
            l_ceb = ((h * el_perp).sum(-1) ** 2).mean()
            loss = loss + self.cmi_beta * l_ceb
            self._leak_log.append(float(l_ceb.detach()))
        elif est == 'adv':
            logits_c = self.critic(grad_reverse(h, self.cmi_beta), E[answers].detach())
            l_ceb = F.cross_entropy(logits_c, ell)   # critic minimizes; student maximizes (GRL)
            loss = loss + l_ceb
            self._leak_log.append(float(l_ceb.detach()))
        elif est == 'club':
            for _ in range(self.cmi_critic_steps):
                lc = self.critic(h.detach(), E[answers].detach())
                Lc = F.cross_entropy(lc, ell)
                self.opt_critic.zero_grad(); Lc.backward(); self.opt_critic.step()
            lc2 = self.critic(h, E[answers].detach())
            logp = F.log_softmax(lc2, dim=-1)
            pos = logp.gather(1, ell[:, None]).squeeze(1)
            logp_ell = logp[:, ell]                                    # [B,B]
            same_y = (answers[:, None] == answers[None, :]).float()
            cnt = same_y.sum(1)
            use_all = (cnt <= 1).float()[:, None]
            w = same_y * (1 - use_all) + torch.ones_like(same_y) * use_all
            neg = (w * logp_ell).sum(1) / w.sum(1)
            l_club = (pos - neg).mean()
            loss = loss + self.cmi_beta * l_club
            self._leak_log.append(float(l_club.detach()))
            self._samey_rate = float((cnt > 1).float().mean())
        return loss

    def train(self, epoch):
        self._leak_log = []
        super().train(epoch)
        if self._leak_log:
            msg = f"CMI leak/CEB epoch {epoch}: {np.mean(self._leak_log):.4f}"
            if self.cmi_estimator == 'club':
                msg += f"  same_y_rate={self._samey_rate:.3f}"
            self.logger.info(msg)

    @torch.no_grad()
    def cmi_finalize(self, csv_path):
        """Post-train metrics (test set): HR@10/NDCG@10, HRLI@1, cos_last, and
        HR@10 by leak tercile (advantaged=low leak / disadvantaged=high leak).
        Uses the in-memory critic (adv/club) or the linear proxy. Writes one CSV row."""
        import csv as _csv
        self.model.eval()
        if self.critic is not None:
            self.critic.eval()
        E = self.model.item_embeddings.weight
        trm = self.args.test_rating_matrix
        leaks, hrli, coslast, rankf = [], [], [], []
        for batch in self.test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            user_ids, input_ids, answers, _, _ = batch
            ell = input_ids[:, -1]
            h = self.model.predict(input_ids, user_ids)[:, -1, :]
            logits = torch.matmul(h, E.transpose(0, 1))
            hrli.append((logits.argmax(1) == ell).float().cpu().numpy())
            coslast.append(F.cosine_similarity(h, E[ell], dim=-1).cpu().numpy())
            if self.cmi_estimator in ('adv', 'club'):
                logp = F.log_softmax(self.critic(h, E[answers]), dim=-1)
                if self.cmi_estimator == 'adv':
                    leak = logp.gather(1, ell[:, None]).squeeze(1)
                else:
                    pos = logp.gather(1, ell[:, None]).squeeze(1)
                    sy = (answers[:, None] == answers[None, :]).float(); cnt = sy.sum(1)
                    ua = (cnt <= 1).float()[:, None]; w = sy * (1 - ua) + torch.ones_like(sy) * ua
                    leak = pos - (w * logp[:, ell]).sum(1) / w.sum(1)
            else:
                ey = F.normalize(E[answers], dim=-1); el = E[ell]
                el_perp = el - (el * ey).sum(-1, keepdim=True) * ey
                leak = (h * el_perp).sum(-1) ** 2
            leaks.append(leak.cpu().numpy())
            rp = logits.cpu().numpy().copy(); bidx = user_ids.cpu().numpy(); y = answers.cpu().numpy()
            try: rp[trm[bidx].toarray() > 0] = 0
            except Exception: rp = rp[:, :-1]; rp[trm[bidx].toarray() > 0] = 0
            rankf.append((rp > rp[np.arange(len(rp)), y][:, None]).sum(1) + 1)
        leaks = np.concatenate(leaks); hrli = np.concatenate(hrli)
        coslast = np.concatenate(coslast); rankf = np.concatenate(rankf)
        hit10 = (rankf <= 10).astype(float)
        ndcg10 = np.where(rankf <= 10, 1.0 / np.log2(rankf + 1), 0.0)
        q1, q2 = np.quantile(leaks, [1/3, 2/3])
        adv_g, dis_g = leaks <= q1, leaks > q2
        row = {
            "dataset": self.args.data_name, "estimator": self.cmi_estimator,
            "beta": self.cmi_beta, "rank_k": self.rank_k, "lambda_pl": self.lambda_kd,
            "HR@10": round(hit10.mean(), 4), "NDCG@10": round(ndcg10.mean(), 4),
            "HRLI@1": round(hrli.mean(), 4), "cos_last": round(float(coslast.mean()), 4),
            "HR@10_adv(lowleak)": round(hit10[adv_g].mean(), 4),
            "HR@10_dis(highleak)": round(hit10[dis_g].mean(), 4),
            "leak_train": round(float(np.mean(self._leak_log)) if self._leak_log else 0.0, 4),
            "samey_rate": round(self._samey_rate, 4) if self.cmi_estimator == 'club' else "",
        }
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            wcsv = _csv.DictWriter(f, fieldnames=list(row.keys()))
            if new: wcsv.writeheader()
            wcsv.writerow(row)
        self.logger.info(f"CMI finalize -> {csv_path}: {row}")
        return row


class DebiasDistillTrainer(CMIDistillTrainer):
    """Gated relative de-bias distillation (kd_mode=debias).

        L = L_rec + lambda_kd * L_PL  (+ lambda_db * L_db   for margin/bpr)

    Replaces the ABSOLUTE repulsion of cmi-linear (shrink h's E_ell component;
    decoupled from prediction -> leak fell but HR fell too) with a RELATIVE
    penalty in the prediction's own score space: only the gap of s_ell vs s_y
    is penalized, gated by w = 1 - cos(E_ell, E_y) (full stop-grad; y==ell ->
    w=0 -> no penalty). Self-limiting via margin / logsigmoid saturation /
    softmax normalization.

    Arms (--debias_mode):
      none         : pure PL baseline (S1 reference; zero extra ops)
      margin       : L_db = (w * relu(m - (cos(h,E_y) - cos(h,E_ell)))).mean()
                     --detach_neg: relu((sg(s_ell)+m) - s_y) ("raise y" only)
      bpr          : L_db = -(w * logsigmoid(s_y_raw - s_ell_raw)).mean(), raw dots,
                     E columns detached (no gradient leak into the embedding table)
      logit_margin : train-only: +w*m on the ell column of the CE logits
                     (L_PL ALWAYS sees the unmodified logits)
      reweight     : control arm, no ell term: CE_i weighted by (1 + gamma*w)

    Warmup (--db_warmup): early E is near-random -> w~1 indiscriminately, so the
    de-bias term is disabled for the first db_warmup epochs.
    Gate embeddings: live student E (detached) by default; --gate_emb_ckpt
    freezes the gate at that checkpoint's item embeddings.
    Eval path is untouched (y / w / modified logits never enter valid/test);
    valid() only ADDs read-only val-HRLI@1 logging. leak (the cmi-linear proxy,
    unchanged definition) is logged for comparability with the beta sweep.
    """

    def __init__(self, student_model, teacher_model,
                 train_dataloader, eval_dataloader, test_dataloader, args, logger):
        args.cmi_estimator = 'none'   # no critic / no CEB term; PL base only
        super().__init__(student_model, teacher_model,
                         train_dataloader, eval_dataloader, test_dataloader, args, logger)
        self.debias_mode = getattr(args, 'debias_mode', 'none')
        self.lambda_db = getattr(args, 'lambda_db', 1.0)
        self.margin_m = getattr(args, 'margin_m', 0.3)
        self.lm_margin = getattr(args, 'lm_margin', 1.0)
        self.gamma_rw = getattr(args, 'gamma_rw', 1.0)
        self.db_warmup = getattr(args, 'db_warmup', 10)
        self.detach_neg = getattr(args, 'detach_neg', False)
        self.gate_E = None
        if getattr(args, 'gate_emb_ckpt', None):
            state = torch.load(args.gate_emb_ckpt, map_location=self.device)
            self.gate_E = state['item_embeddings.weight'].to(self.device)
            logger.info(f"DEBIAS gate frozen from {args.gate_emb_ckpt} "
                        f"(shape {tuple(self.gate_E.shape)})")
        self._cur_epoch = 0
        self._ep = {}
        self._epoch_csv = os.path.join(args.output_dir, '..', 'results',
                                       f"{args.train_name}_epochs.csv")
        os.makedirs(os.path.dirname(self._epoch_csv), exist_ok=True)
        logger.info(f"DEBIAS config: mode={self.debias_mode}, lambda_db={self.lambda_db}, "
                    f"margin_m={self.margin_m}, lm_margin={self.lm_margin}, "
                    f"gamma_rw={self.gamma_rw}, warmup={self.db_warmup}, "
                    f"detach_neg={self.detach_neg}, base=PL(lam={self.lambda_kd},k={self.rank_k})")

    def _gate(self, ell, answers):
        """w = 1 - cos(E_ell, E_y), full stop-grad."""
        with torch.no_grad():
            Eg = self.gate_E if self.gate_E is not None else self.model.item_embeddings.weight
            e_l_n = F.normalize(Eg[ell], dim=-1)
            e_y_n = F.normalize(Eg[answers], dim=-1)
            return (1.0 - (e_l_n * e_y_n).sum(-1)).clamp(0.0, 1.0)

    def _acc(self, key, val, n=1):
        s, c = self._ep.get(key, (0.0, 0))
        self._ep[key] = (s + float(val) * n, c + n)

    def _compute_train_loss(self, batch):
        user_ids, input_ids, answers, neg_answer, same_target = batch
        ell = input_ids[:, -1]
        # Length-1 train prefixes have an ALL-PADDING input (ell==0): there is no
        # last item, so they get w=0 (no penalty) and are excluded from diagnostics.
        valid_l = (ell != 0)
        E = self.model.item_embeddings.weight
        h = self.model.predict(input_ids, user_ids)[:, -1, :]              # [B, d]
        logits = torch.matmul(h, E.transpose(0, 1))                        # [B, V]
        B = logits.size(0)
        w = self._gate(ell, answers) * valid_l.float()                     # [B], sg
        active = (self._cur_epoch >= self.db_warmup)
        mode = self.debias_mode

        # ----- L_rec (logit_margin / reweight modify it; others use plain CE) -----
        if mode == 'logit_margin' and active and self.lm_margin != 0.0:
            logits_ce = logits.clone()
            logits_ce[torch.arange(B, device=logits.device), ell] += w * self.lm_margin
            l_rec = F.cross_entropy(logits_ce, answers)
        elif mode == 'reweight' and active and self.gamma_rw != 0.0:
            ce_i = F.cross_entropy(logits, answers, reduction='none')
            l_rec = ((1.0 + self.gamma_rw * w) * ce_i).mean()
        else:
            l_rec = F.cross_entropy(logits, answers)

        # ----- L_PL: ALWAYS the unmodified logits (S3) -----
        z_ord = self._teacher_logits(input_ids, user_ids)
        l_pl = self._pl_term(logits, z_ord)
        loss = l_rec + self.lambda_kd * l_pl

        # ----- L_db (margin / bpr only) -----
        l_db = None
        if mode == 'margin' and active and self.lambda_db != 0.0:
            h_n = F.normalize(h, dim=-1)                                   # grad flows via h
            with torch.no_grad():
                e_l_s = F.normalize(E[ell], dim=-1)
                e_y_s = F.normalize(E[answers], dim=-1)
            s_y_c = (h_n * e_y_s).sum(-1)
            s_l_c = (h_n * e_l_s).sum(-1)
            if self.detach_neg:
                l_db = (w * F.relu((s_l_c.detach() + self.margin_m) - s_y_c)).mean()
            else:
                l_db = (w * F.relu(self.margin_m - (s_y_c - s_l_c))).mean()
            loss = loss + self.lambda_db * l_db
        elif mode == 'bpr' and active and self.lambda_db != 0.0:
            s_y_r = (h * E[answers].detach()).sum(-1)
            s_l_r = (h * E[ell].detach()).sum(-1)
            l_db = -(w * F.logsigmoid(s_y_r - s_l_r)).mean()
            loss = loss + self.lambda_db * l_db

        # ----- per-epoch diagnostics (no_grad; §5 of the spec; valid rows only) -----
        with torch.no_grad():
            hi, lo = (w > 0.5) & valid_l, (w < 0.2) & valid_l
            sy_r = (h * E[answers]).sum(-1)
            sl_r = (h * E[ell]).sum(-1)
            h_nn = F.normalize(h, dim=-1)
            sy_c = (h_nn * F.normalize(E[answers], dim=-1)).sum(-1)
            sl_c = (h_nn * F.normalize(E[ell], dim=-1)).sum(-1)
            ey = F.normalize(E[answers], dim=-1)
            el = E[ell]
            el_perp = el - (el * ey).sum(-1, keepdim=True) * ey
            leak = ((h * el_perp).sum(-1) ** 2)                            # cmi-linear proxy
            self._acc('l_rec', l_rec, 1); self._acc('l_pl', l_pl, 1)
            if l_db is not None:
                self._acc('l_db', l_db, 1)
            nv = int(valid_l.sum())
            if nv > 0:
                self._acc('w_mean', w[valid_l].mean(), nv)
                self._acc('w_frac_gt05', hi.float().sum() / nv, nv)
                self._acc('w_frac_lt02', lo.float().sum() / nv, nv)
            self._acc('pad_frac', 1.0 - nv / max(B, 1), 1)
            self._acc('leak', leak.mean(), 1)
            for tag, g in (('all', valid_l), ('hi', hi), ('lo', lo)):
                for nm, v in (('s_y_raw', sy_r), ('s_l_raw', sl_r),
                              ('s_y_cos', sy_c), ('s_l_cos', sl_c)):
                    vv = v[g]
                    if vv.numel() > 0:
                        self._acc(f'{nm}_{tag}', vv.mean(), int(vv.numel()))
            hist = torch.histc(w[valid_l], bins=10, min=0.0, max=1.0)
            self._ep['w_hist'] = self._ep.get('w_hist', torch.zeros(10)) + hist.cpu()
        return loss

    def train(self, epoch):
        self._cur_epoch = epoch
        self._ep = {}
        if epoch == self.db_warmup:
            self.logger.info(f"DEBIAS term ACTIVATED at epoch {epoch}")
        Trainer.train(self, epoch)                                         # plain loop
        m = {k: (v[0] / v[1] if v[1] else float('nan'))
             for k, v in self._ep.items() if k != 'w_hist'}
        self.logger.info(
            f"DEBIAS epoch {epoch} active={epoch >= self.db_warmup} "
            f"L_rec={m.get('l_rec', float('nan')):.4f} L_PL={m.get('l_pl', float('nan')):.4f} "
            f"L_db={m.get('l_db', float('nan')):.4f} w_mean={m.get('w_mean', float('nan')):.3f} "
            f"w>.5={m.get('w_frac_gt05', float('nan')):.3f} "
            f"w<.2={m.get('w_frac_lt02', float('nan')):.3f} "
            f"s_y_raw(hi)={m.get('s_y_raw_hi', float('nan')):.4f} "
            f"s_l_raw(hi)={m.get('s_l_raw_hi', float('nan')):.4f} "
            f"leak={m.get('leak', float('nan')):.4f}")
        self._ep_means = m

    @torch.no_grad()
    def _val_hrli1(self):
        """Read-only: fraction of val instances whose unmasked top-1 == last input item."""
        self.model.eval()
        E = self.model.item_embeddings.weight
        n_hit, n = 0, 0
        for batch in self.eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            user_ids, input_ids, _, _, _ = batch
            h = self.model.predict(input_ids, user_ids)[:, -1, :]
            top1 = torch.matmul(h, E.transpose(0, 1)).argmax(1)
            n_hit += int((top1 == input_ids[:, -1]).sum())
            n += input_ids.size(0)
        return n_hit / max(n, 1)

    def valid(self, epoch):
        scores, info = super().valid(epoch)                                # protocol untouched
        if hasattr(self, '_ep_means'):                                     # training loop only
            import csv as _csv
            m = self._ep_means
            row = {'epoch': epoch, 'active': int(epoch >= self.db_warmup)}
            for k in ('l_rec', 'l_pl', 'l_db', 'w_mean', 'w_frac_gt05', 'w_frac_lt02',
                      's_y_raw_all', 's_y_raw_hi', 's_y_raw_lo',
                      's_l_raw_all', 's_l_raw_hi', 's_l_raw_lo',
                      's_y_cos_all', 's_y_cos_hi', 's_y_cos_lo',
                      's_l_cos_all', 's_l_cos_hi', 's_l_cos_lo', 'leak'):
                row[k] = round(m.get(k, float('nan')), 6)
            row['val_HR10'] = scores[2]
            row['val_NDCG10'] = scores[3]
            row['val_HRLI1'] = round(self._val_hrli1(), 4)
            wh = self._ep.get('w_hist', torch.zeros(10))
            wh = (wh / wh.sum()).tolist() if wh.sum() > 0 else [float('nan')] * 10
            for i, v in enumerate(wh):
                row[f'w_hist{i}'] = round(v, 4)
            new = not os.path.exists(self._epoch_csv)
            with open(self._epoch_csv, 'a', newline='') as f:
                wcsv = _csv.DictWriter(f, fieldnames=list(row.keys()))
                if new: wcsv.writeheader()
                wcsv.writerow(row)
        return scores, info

    @torch.no_grad()
    def debias_finalize(self, csv_path):
        """Test-set metrics: HR/NDCG/HRLI/cos_last/leak + HR@10 by w-tercile
        (lo = advantaged group; over-removal watch) + HR@10 by item-popularity
        tercile of the TARGET y (train-sequence occurrence counts; popularity-
        debias confound check). One CSV row + log line."""
        import csv as _csv
        from dataset import get_seq_dic
        self.model.eval()
        E = self.model.item_embeddings.weight
        trm = self.args.test_rating_matrix
        # item popularity = occurrence count in the train part (seq[:-2]) of each user
        seq_dic, _, _ = get_seq_dic(self.args)
        pop = np.zeros(self.args.item_size, dtype=np.int64)
        for seq in seq_dic['user_seq']:
            for it in seq[:-2]:
                pop[it] += 1
        ws, hrli, coslast, leaks, rankf, ypop = [], [], [], [], [], []
        for batch in self.test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            user_ids, input_ids, answers, _, _ = batch
            ell = input_ids[:, -1]
            h = self.model.predict(input_ids, user_ids)[:, -1, :]
            logits = torch.matmul(h, E.transpose(0, 1))
            ws.append(self._gate(ell, answers).cpu().numpy())
            hrli.append((logits.argmax(1) == ell).float().cpu().numpy())
            coslast.append(F.cosine_similarity(h, E[ell], dim=-1).cpu().numpy())
            ey = F.normalize(E[answers], dim=-1); el = E[ell]
            el_perp = el - (el * ey).sum(-1, keepdim=True) * ey
            leaks.append(((h * el_perp).sum(-1) ** 2).cpu().numpy())
            rp = logits.cpu().numpy().copy()
            bidx = user_ids.cpu().numpy(); y = answers.cpu().numpy()
            try: rp[trm[bidx].toarray() > 0] = 0
            except Exception: rp = rp[:, :-1]; rp[trm[bidx].toarray() > 0] = 0
            rankf.append((rp > rp[np.arange(len(rp)), y][:, None]).sum(1) + 1)
            ypop.append(pop[y])
        ws = np.concatenate(ws); hrli = np.concatenate(hrli)
        coslast = np.concatenate(coslast); leaks = np.concatenate(leaks)
        rankf = np.concatenate(rankf); ypop = np.concatenate(ypop)
        hit10 = (rankf <= 10).astype(float)
        ndcg10 = np.where(rankf <= 10, 1.0 / np.log2(rankf + 1), 0.0)
        wq1, wq2 = np.quantile(ws, [1 / 3, 2 / 3])
        w_lo, w_hi = ws <= wq1, ws > wq2
        w_mid = ~w_lo & ~w_hi
        pq1, pq2 = np.quantile(ypop, [1 / 3, 2 / 3])
        p_lo, p_hi = ypop <= pq1, ypop > pq2
        p_mid = ~p_lo & ~p_hi
        m = getattr(self, '_ep_means', {})
        row = {
            "dataset": self.args.data_name, "mode": self.debias_mode,
            "lambda_db": self.lambda_db, "margin_m": self.margin_m,
            "lm_margin": self.lm_margin, "gamma_rw": self.gamma_rw,
            "warmup": self.db_warmup, "detach_neg": int(self.detach_neg),
            "rank_k": self.rank_k, "lambda_pl": self.lambda_kd,
            "HR@10": round(hit10.mean(), 4), "NDCG@10": round(ndcg10.mean(), 4),
            "HRLI@1": round(hrli.mean(), 4), "cos_last": round(float(coslast.mean()), 4),
            "leak_test": round(float(leaks.mean()), 4),
            "HR@10_w_lo": round(hit10[w_lo].mean(), 4),
            "HR@10_w_mid": round(hit10[w_mid].mean(), 4) if w_mid.any() else "",
            "HR@10_w_hi": round(hit10[w_hi].mean(), 4),
            "HR@10_pop_lo": round(hit10[p_lo].mean(), 4),
            "HR@10_pop_mid": round(hit10[p_mid].mean(), 4) if p_mid.any() else "",
            "HR@10_pop_hi": round(hit10[p_hi].mean(), 4),
            "s_y_raw_hi_final": round(m.get('s_y_raw_hi', float('nan')), 4),
            "s_l_raw_hi_final": round(m.get('s_l_raw_hi', float('nan')), 4),
            "w_test_mean": round(float(ws.mean()), 4),
        }
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            wcsv = _csv.DictWriter(f, fieldnames=list(row.keys()))
            if new: wcsv.writeheader()
            wcsv.writerow(row)
        self.logger.info(f"DEBIAS finalize -> {csv_path}: {row}")
        return row


class RepDistillTrainer(CMIDistillTrainer):
    """kappa-corrected relational distillation (kd_mode=rep).

        L = L_rec + lambda_kd * L_PL + lambda_rep * L_rep

    L_rep matches the STUDENT's cosine-similarity structure S = hs_hat @ hs_hat.T
    to a TEACHER structure T (off-diagonal MSE). Dimension-free (S, T are scalar
    matrices), so d_s != d_t works unchanged. kappa = d*(1-a) is the teacher's
    per-instance harmful-dominance score, computed ON THE FLY inside the same
    no_grad teacher forward already required for PL (zero extra forward cost):
        d = cos(h_t, E_t[ell]).clamp(0,1), a = cos(E_t[ell], E_t[y]).clamp(0,1)
        g = h_t - min(kappa, .95) * (h_t . e_l) * e_l   (rank-1 soft removal)

    rep_mode:
      none          : L_rep = 0 (pure PL baseline, S1 reference)
      corrected     : T from g_hat (arm A; D3 PASS form)
      raw           : T from h_t_hat (arm A0, the decisive ablation)
      pairgate      : T from h_t_hat, pair weights (1-k_i)(1-k_j) (D3-FAIL fallback A)
      shuffled      : corrected with kappa permuted within batch (placebo C)
      pairgate_shuf : pairgate with kappa permuted within batch (placebo C)

    Teacher quantities are inside no_grad (frozen teacher) -> structurally
    stop-grad (asserted, S2). ell==0 rows (length-1 train prefixes) get kappa=0
    (no correction / weight 1). Eval path untouched; leak (cmi-linear proxy,
    unchanged definition) logged for comparability.
    """

    def __init__(self, student_model, teacher_model,
                 train_dataloader, eval_dataloader, test_dataloader, args, logger):
        args.cmi_estimator = 'none'
        super().__init__(student_model, teacher_model,
                         train_dataloader, eval_dataloader, test_dataloader, args, logger)
        self.rep_mode = getattr(args, 'rep_mode', 'none')
        self.lambda_rep = getattr(args, 'lambda_rep', 0.0)
        self._ep = {}
        self._epoch_csv = os.path.join(args.output_dir, '..', 'results',
                                       f"{args.train_name}_epochs.csv")
        os.makedirs(os.path.dirname(self._epoch_csv), exist_ok=True)
        logger.info(f"REP config: mode={self.rep_mode}, lambda_rep={self.lambda_rep}, "
                    f"base=PL(lam={self.lambda_kd},k={self.rank_k})")

    def _acc(self, key, val, n=1):
        s, c = self._ep.get(key, (0.0, 0))
        self._ep[key] = (s + float(val) * n, c + n)

    @torch.no_grad()
    def _teacher_quant(self, input_ids, answers):
        """h_t, z_ord (PL target), kappa, t_hat (T row vectors per rep_mode)."""
        E_t = self.teacher.item_embeddings.weight
        h_t = self.teacher.predict(input_ids, None)[:, -1, :]
        z_ord = torch.matmul(h_t, E_t.transpose(0, 1))          # == _teacher_logits
        ell = input_ids[:, -1]
        e_l = F.normalize(E_t[ell], dim=-1)
        e_y = F.normalize(E_t[answers], dim=-1)
        d = (F.normalize(h_t, dim=-1) * e_l).sum(-1).clamp(0, 1)
        a = (e_l * e_y).sum(-1).clamp(0, 1)
        kappa = d * (1 - a) * (ell != 0).float()
        k_eff = kappa
        if self.rep_mode in ('shuffled', 'pairgate_shuf'):
            k_eff = kappa[torch.randperm(kappa.size(0), device=kappa.device)]
        if self.rep_mode in ('corrected', 'shuffled'):
            p = (h_t * e_l).sum(-1, keepdim=True)
            g = h_t - k_eff.clamp(max=0.95).unsqueeze(-1) * p * e_l
            t_hat = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:                                                   # raw / pairgate(+shuf)
            t_hat = F.normalize(h_t, dim=-1)
        return z_ord, kappa, k_eff, t_hat

    def _compute_train_loss(self, batch):
        user_ids, input_ids, answers, neg_answer, same_target = batch
        E_s = self.model.item_embeddings.weight
        h_s = self.model.predict(input_ids, user_ids)[:, -1, :]
        logits = torch.matmul(h_s, E_s.transpose(0, 1))
        l_rec = F.cross_entropy(logits, answers)
        z_ord, kappa, k_eff, t_hat = self._teacher_quant(input_ids, answers)
        l_pl = self._pl_term(logits, z_ord)                     # unmodified logits (S5)
        loss = l_rec + self.lambda_kd * l_pl

        l_rep = None
        if self.rep_mode != 'none' and self.lambda_rep != 0.0:
            assert not t_hat.requires_grad and not k_eff.requires_grad   # S2
            B = h_s.size(0)
            hs_hat = F.normalize(h_s, dim=-1)
            S = hs_hat @ hs_hat.t()
            T = t_hat @ t_hat.t()
            mask = ~torch.eye(B, dtype=torch.bool, device=S.device)
            if self.rep_mode in ('pairgate', 'pairgate_shuf'):
                w = 1 - k_eff
                M = (w.unsqueeze(0) * w.unsqueeze(1))[mask]
                l_rep = (M * ((S - T)[mask] ** 2)).sum() / M.sum().clamp_min(1e-6)
            else:
                l_rep = ((S - T)[mask] ** 2).mean()
            loss = loss + self.lambda_rep * l_rep

        # ----- per-epoch diagnostics (no_grad) -----
        with torch.no_grad():
            self._acc('l_rec', l_rec, 1); self._acc('l_pl', l_pl, 1)
            if l_rep is not None:
                self._acc('l_rep', l_rep, 1)
            self._acc('kappa_mean', kappa.mean(), 1)
            if self.rep_mode != 'none':
                B = h_s.size(0)
                hs_hat = F.normalize(h_s, dim=-1)
                S = (hs_hat @ hs_hat.t())
                T = (t_hat @ t_hat.t())
                mask = ~torch.eye(B, dtype=torch.bool, device=S.device)
                s_off, t_off = S[mask], T[mask]
                def _corr(x, yv):
                    if x.numel() < 3:
                        return float('nan')
                    xc, yc = x - x.mean(), yv - yv.mean()
                    return float((xc * yc).sum() /
                                 (xc.norm() * yc.norm()).clamp_min(1e-9))
                self._acc('corr_all', _corr(s_off, t_off), 1)
                hi_i = kappa > 0.5
                lo_i = kappa < 0.2
                hi_pair = (hi_i.unsqueeze(0) & hi_i.unsqueeze(1))[mask]
                lo_pair = (lo_i.unsqueeze(0) & lo_i.unsqueeze(1))[mask]
                if hi_pair.sum() >= 3:
                    self._acc('corr_hik', _corr(s_off[hi_pair], t_off[hi_pair]), 1)
                if lo_pair.sum() >= 3:
                    self._acc('corr_lok', _corr(s_off[lo_pair], t_off[lo_pair]), 1)
            ell = input_ids[:, -1]
            ey = F.normalize(E_s[answers], dim=-1)
            el = E_s[ell]
            el_perp = el - (el * ey).sum(-1, keepdim=True) * ey
            self._acc('leak', ((h_s * el_perp).sum(-1) ** 2).mean(), 1)
        return loss

    def train(self, epoch):
        self._ep = {}
        Trainer.train(self, epoch)
        m = {k: (v[0] / v[1] if v[1] else float('nan')) for k, v in self._ep.items()}
        self.logger.info(
            f"REP epoch {epoch} L_rec={m.get('l_rec', float('nan')):.4f} "
            f"L_PL={m.get('l_pl', float('nan')):.4f} L_rep={m.get('l_rep', float('nan')):.6f} "
            f"corr(S,T)={m.get('corr_all', float('nan')):.4f} "
            f"corr_hik={m.get('corr_hik', float('nan')):.4f} "
            f"corr_lok={m.get('corr_lok', float('nan')):.4f} "
            f"kappa={m.get('kappa_mean', float('nan')):.3f} "
            f"leak={m.get('leak', float('nan')):.4f}")
        self._ep_means = m

    def valid(self, epoch):
        scores, info = super().valid(epoch)
        if hasattr(self, '_ep_means'):
            import csv as _csv
            m = self._ep_means
            row = {'epoch': epoch}
            for k in ('l_rec', 'l_pl', 'l_rep', 'corr_all', 'corr_hik', 'corr_lok',
                      'kappa_mean', 'leak'):
                row[k] = round(m.get(k, float('nan')), 6)
            row['val_HR10'] = scores[2]
            row['val_NDCG10'] = scores[3]
            new = not os.path.exists(self._epoch_csv)
            with open(self._epoch_csv, 'a', newline='') as f:
                wcsv = _csv.DictWriter(f, fieldnames=list(row.keys()))
                if new: wcsv.writeheader()
                wcsv.writerow(row)
        return scores, info

    @torch.no_grad()
    def rep_finalize(self, csv_path):
        """Test metrics + kappa-tercile HR (is the gain concentrated in high kappa?)
        + popularity-tercile HR. One CSV row + log line."""
        import csv as _csv
        from dataset import get_seq_dic
        self.model.eval()
        E_s = self.model.item_embeddings.weight
        trm = self.args.test_rating_matrix
        seq_dic, _, _ = get_seq_dic(self.args)
        pop = np.zeros(self.args.item_size, dtype=np.int64)
        for seq in seq_dic['user_seq']:
            for it in seq[:-2]:
                pop[it] += 1
        kaps, hrli, coslast, leaks, rankf, ypop = [], [], [], [], [], []
        for batch in self.test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            user_ids, input_ids, answers, _, _ = batch
            ell = input_ids[:, -1]
            h = self.model.predict(input_ids, user_ids)[:, -1, :]
            logits = torch.matmul(h, E_s.transpose(0, 1))
            _, kappa, _, _ = self._teacher_quant(input_ids, answers)
            kaps.append(kappa.cpu().numpy())
            hrli.append((logits.argmax(1) == ell).float().cpu().numpy())
            coslast.append(F.cosine_similarity(h, E_s[ell], dim=-1).cpu().numpy())
            ey = F.normalize(E_s[answers], dim=-1); el = E_s[ell]
            el_perp = el - (el * ey).sum(-1, keepdim=True) * ey
            leaks.append(((h * el_perp).sum(-1) ** 2).cpu().numpy())
            rp = logits.cpu().numpy().copy()
            bidx = user_ids.cpu().numpy(); y = answers.cpu().numpy()
            try: rp[trm[bidx].toarray() > 0] = 0
            except Exception: rp = rp[:, :-1]; rp[trm[bidx].toarray() > 0] = 0
            rankf.append((rp > rp[np.arange(len(rp)), y][:, None]).sum(1) + 1)
            ypop.append(pop[y])
        kaps = np.concatenate(kaps); hrli = np.concatenate(hrli)
        coslast = np.concatenate(coslast); leaks = np.concatenate(leaks)
        rankf = np.concatenate(rankf); ypop = np.concatenate(ypop)
        hit10 = (rankf <= 10).astype(float)
        ndcg10 = np.where(rankf <= 10, 1.0 / np.log2(rankf + 1), 0.0)
        kq1, kq2 = np.quantile(kaps, [1 / 3, 2 / 3])
        k_lo, k_hi = kaps <= kq1, kaps > kq2
        k_mid = ~k_lo & ~k_hi
        pq1, pq2 = np.quantile(ypop, [1 / 3, 2 / 3])
        p_lo, p_hi = ypop <= pq1, ypop > pq2
        p_mid = ~p_lo & ~p_hi
        m = getattr(self, '_ep_means', {})
        row = {
            "dataset": self.args.data_name, "mode": self.rep_mode,
            "lambda_rep": self.lambda_rep,
            "rank_k": self.rank_k, "lambda_pl": self.lambda_kd,
            "HR@10": round(hit10.mean(), 4), "NDCG@10": round(ndcg10.mean(), 4),
            "HRLI@1": round(hrli.mean(), 4), "cos_last": round(float(coslast.mean()), 4),
            "leak_test": round(float(leaks.mean()), 4),
            "HR@10_k_lo": round(hit10[k_lo].mean(), 4),
            "HR@10_k_mid": round(hit10[k_mid].mean(), 4) if k_mid.any() else "",
            "HR@10_k_hi": round(hit10[k_hi].mean(), 4),
            "HR@10_pop_lo": round(hit10[p_lo].mean(), 4),
            "HR@10_pop_mid": round(hit10[p_mid].mean(), 4) if p_mid.any() else "",
            "HR@10_pop_hi": round(hit10[p_hi].mean(), 4),
            "corr_ST_final": round(m.get('corr_all', float('nan')), 4),
        }
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            wcsv = _csv.DictWriter(f, fieldnames=list(row.keys()))
            if new: wcsv.writeheader()
            wcsv.writerow(row)
        self.logger.info(f"REP finalize -> {csv_path}: {row}")
        return row
