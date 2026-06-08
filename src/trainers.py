import math
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
