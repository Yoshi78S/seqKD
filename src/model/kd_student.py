import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import FeedForward, LayerNorm


class StudentBlock(nn.Module):
    """StudentBlock = GRU [+ LayerNorm] [+ FFN] with per-component toggles.

    Ablation flags (all on `args`, default False):
        abl_no_block_ln : drop LayerNorm after the GRU
        abl_no_ffn      : drop the FeedForward sub-module
    """
    def __init__(self, args):
        super().__init__()
        self.use_block_ln = not getattr(args, 'abl_no_block_ln', False)
        self.use_ffn = not getattr(args, 'abl_no_ffn', False)

        self.gru = nn.GRU(
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.gru_dropout = nn.Dropout(args.hidden_dropout_prob)
        if self.use_block_ln:
            self.gru_layernorm = LayerNorm(args.hidden_size, eps=1e-12)
        if self.use_ffn:
            self.feed_forward = FeedForward(args)

    def forward(self, hidden_states):
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(hidden_states)
        gru_out = self.gru_dropout(gru_out)
        if self.use_block_ln:
            gru_out = self.gru_layernorm(gru_out)
        if self.use_ffn:
            gru_out = self.feed_forward(gru_out)
        return gru_out


class StudentEncoder(nn.Module):
    """StudentBlock × N, or a single multi-layer GRU when `abl_flat_gru`."""
    def __init__(self, args):
        super().__init__()
        self.flat = getattr(args, 'abl_flat_gru', False)
        if self.flat:
            self.gru = nn.GRU(
                input_size=args.hidden_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_hidden_layers,
                batch_first=True,
                bias=False,
            )
            self.dropout = nn.Dropout(args.hidden_dropout_prob)
        else:
            self.blocks = nn.ModuleList([
                StudentBlock(args) for _ in range(args.num_hidden_layers)
            ])

    def forward(self, hidden_states):
        if self.flat:
            self.gru.flatten_parameters()
            out, _ = self.gru(hidden_states)
            return self.dropout(out)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class KDStudentModel(SequentialRecModel):
    """Configurable KDStudent.

    Ablation flags (all on `args`, default False):
        abl_no_pos_emb  : skip position embeddings (item_emb only)
        abl_no_input_ln : skip the input LayerNorm + dropout
        abl_no_ffn      : drop block-internal FFN
        abl_no_block_ln : drop block-internal LayerNorm
        abl_flat_gru    : replace block stack with one nn.GRU(num_layers=N)
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.no_pos_emb = getattr(args, 'abl_no_pos_emb', False)
        self.no_input_ln = getattr(args, 'abl_no_input_ln', False)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.student_encoder = StudentEncoder(args)
        self.apply(self.init_weights)

    def _embed(self, input_ids):
        sequence_emb = self.item_embeddings(input_ids)
        if not self.no_pos_emb:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            sequence_emb = sequence_emb + self.position_embeddings(position_ids)
        if not self.no_input_ln:
            sequence_emb = self.LayerNorm(sequence_emb)
            sequence_emb = self.dropout(sequence_emb)
        return sequence_emb

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self._embed(input_ids)
        output = self.student_encoder(sequence_emb)
        return output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)[:, -1, :]
        logits = torch.matmul(seq_output, self.item_embeddings.weight.T)
        return nn.CrossEntropyLoss()(logits, answers)

    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        return self.forward(input_ids)
