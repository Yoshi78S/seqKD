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
            self.teacher_extractor.register(
                f'layer_{i}', block.layer.attention_layer.dense, detach=True)

        self.student_extractor = HiddenStateExtractor()
        for i, block in enumerate(self.model.student_encoder.blocks):
            self.student_extractor.register(
                f'layer_{i}', block.gru, detach=False)

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

        logger.info(f"KDStudent HS-KD config: lambda_hs={self.lambda_hs}, "
                     f"loss={args.hs_loss_type}, pos={args.hs_position_mode}, "
                     f"layer={self.hs_layer_mode}")

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

        return l_rec + self.lambda_kd * l_pred + self.lambda_hs * l_hs
