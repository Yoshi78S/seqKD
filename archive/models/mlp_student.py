import torch
import torch.nn as nn
import copy
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward

"""
[Description]
A minimal MLP-based student model for sequential recommendation.

The encoder is a stack of position-wise FeedForward blocks (no attention,
no FFT) operating on `item_emb + position_emb` representations. The model
deliberately avoids any sequence-mixing operation that depends on sequence
length (Mamba/SSM, attention, FFT) so its behaviour stays stable for short
histories — the regime where Mamba teachers struggle.

Output is `[batch, seq_len, hidden]` to stay compatible with the existing
BSARec trainer / evaluation pipeline.
"""


class MLPStudentEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        block = FeedForward(args)
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(args.num_hidden_layers)]
        )

    def forward(self, hidden_states, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class MLPStudentModel(SequentialRecModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = MLPStudentEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(
            sequence_emb, output_all_encoded_layers=True
        )
        if all_sequence_output:
            return item_encoded_layers
        return item_encoded_layers[-1]

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_out = self.forward(input_ids)
        seq_out = seq_out[:, -1, :]
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return nn.CrossEntropyLoss()(logits, answers)
