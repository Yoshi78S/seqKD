import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm

"""
[Paper]
Author: Zhenrui Yue et al.
Title: "Linear Recurrent Units for Sequential Recommendation."
Conference: WSDM 2024

[Code Reference]
https://github.com/yueeeeeeee/LRURec
"""


class LRULayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, r_min=0.8, r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model

        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=True).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=True).to(torch.cfloat)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)
        mask_ = mask.reshape(B * L // l, l)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]
        if i > 1:
            lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma

        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        x = self.dropout(self.out_proj(h).real) + x
        return self.layer_norm(x)


class LRUPositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)


class LRUBlock(nn.Module):
    def __init__(self, d_model, dropout, attn_dropout):
        super().__init__()
        self.lru_layer = LRULayer(d_model=d_model, dropout=attn_dropout)
        self.feed_forward = LRUPositionwiseFeedForward(d_model=d_model, d_ff=d_model * 4, dropout=dropout)

    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        return self.feed_forward(x)


class LRURecModel(SequentialRecModel):
    def __init__(self, args):
        super(LRURecModel, self).__init__(args)
        self.args = args

        self.LayerNorm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.lru_blocks = nn.ModuleList([
            LRUBlock(
                d_model=args.hidden_size,
                dropout=args.hidden_dropout_prob,
                attn_dropout=args.attention_probs_dropout_prob,
            )
            for _ in range(args.num_hidden_layers)
        ])

        self._truncated_normal_init()

    def _truncated_normal_init(self, mean=0.0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
            for n, p in self.named_parameters():
                if "layer_norm" in n or "LayerNorm" in n or "params_log" in n:
                    continue
                if torch.is_complex(p):
                    p.real.uniform_(2 * l - 1, 2 * u - 1)
                    p.imag.uniform_(2 * l - 1, 2 * u - 1)
                    p.real.erfinv_()
                    p.imag.erfinv_()
                    p.real.mul_(std * math.sqrt(2.))
                    p.imag.mul_(std * math.sqrt(2.))
                    p.real.add_(mean)
                    p.imag.add_(mean)
                else:
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def _embed(self, input_ids):
        mask = input_ids > 0
        x = self.item_embeddings(input_ids)
        x = self.dropout(x)
        x = self.LayerNorm(x)
        return x, mask

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        x, mask = self._embed(input_ids)

        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(max(seq_len, 2))))
        target_len = 2 ** log2_L
        pad_len = target_len - seq_len
        x = F.pad(x, (0, 0, pad_len, 0, 0, 0))
        mask_ = F.pad(mask, (pad_len, 0, 0, 0))

        outputs = []
        for block in self.lru_blocks:
            x = block(x, mask_)
            outputs.append(x[:, -seq_len:])

        if all_sequence_output:
            return outputs
        return outputs[-1]

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_out = self.forward(input_ids)
        seq_out = seq_out[:, -1, :]
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return nn.CrossEntropyLoss()(logits, answers)
