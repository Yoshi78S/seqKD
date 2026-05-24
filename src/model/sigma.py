import torch
import torch.nn as nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm

"""
[Paper]
Author: Ziwei Liu et al.
Title: "Bidirectional Gated Mamba for Sequential Recommendation."
arXiv 2408.11451 (2024) — also known as SIGMA: Selective Gated Mamba.

[Code Reference]
https://github.com/ziwliu-cityu/SIMGA
"""

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
    _MAMBA_IMPORT_ERR = None
except Exception as _e:
    Mamba = None
    _MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERR = _e


class SIGMAFeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x):
        h = self.w_1(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.w_2(h)
        h = self.dropout(h)
        return self.LayerNorm(h + x)


class SIGMAGatedMambaBlock(nn.Module):
    """Gated Mamba Block (GMambaBlock) from SIGMA.

    Mixes three streams with learnable softmax weights:
      - forward Mamba on x
      - backward (partially flipped) Mamba on flipped x
      - GRU on a 1D-conv view of x
    """

    def __init__(self, d_model, d_state, d_conv, expand, max_seq_length):
        super().__init__()
        self.combining_weights = nn.Parameter(
            torch.tensor([0.1, 0.1, 0.8], dtype=torch.float32)
        )
        self.dense1 = nn.Linear(d_model, d_model)
        self.dense2 = nn.Linear(d_model, d_model)
        self.projection = nn.Linear(d_model, d_model)

        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gru = nn.GRU(d_model, d_model, num_layers=1, bias=False, batch_first=True)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

        # Selective gates
        self.selective_gate_sig = nn.Sequential(nn.Sigmoid(), nn.Linear(d_model, d_model))
        self.selective_gate_si = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model))
        self.selective_gate = nn.Dropout(0.2)

        # Reverse the leading positions; keep the most-recent tail unchanged.
        # Original implementation hard-codes 45 (= max_seq_length=50 minus 5).
        self.flip_n = max(max_seq_length - 5, 1)

    def forward(self, x):
        self.gru.flatten_parameters()
        weights = F.softmax(self.combining_weights.data, dim=0)

        h1 = self.dense1(x)

        # 1d-conv view used by GRU stream
        g1 = self.conv1d(x.transpose(1, 2)).transpose(1, 2)

        # Partially flipped sequence (preserves last 5 positions)
        flipped = x.clone()
        flipped[:, :self.flip_n, :] = x[:, :self.flip_n, :].flip(dims=[1])
        h2 = self.dense2(flipped) + flipped

        mamba_fwd = self.mamba(x)
        mamba_bwd = self.mamba(flipped)

        h1 = x + h1
        # NOTE: faithfully matches reference code where h2's selective gate
        # mixes silu(h2) with sigmoid(h1) (not h2). This is intentional in the
        # released implementation.
        h1 = self.selective_gate_si(h1) + self.selective_gate_sig(h1)
        h1 = self.selective_gate(h1)
        h2 = self.selective_gate_si(h2) + self.selective_gate_sig(h1)
        h2 = self.selective_gate(h2)

        mamba_fwd = mamba_fwd * h1 + mamba_fwd
        mamba_bwd = mamba_bwd * h2 + mamba_bwd

        gru_output, _ = self.gru(g1)

        combined = (
            weights[2] * mamba_fwd
            + weights[1] * mamba_bwd
            + weights[0] * gru_output
        )
        return self.projection(combined)


class SIGMABlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers, max_seq_length):
        super().__init__()
        self.num_layers = num_layers
        self.gmamba = SIGMAGatedMambaBlock(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            max_seq_length=max_seq_length,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = SIGMAFeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, x):
        h = self.gmamba(x)
        if self.num_layers == 1:
            h = self.LayerNorm(self.dropout(h))
        else:
            h = self.LayerNorm(self.dropout(h) + x)
        return self.ffn(h)


class SIGMAModel(SequentialRecModel):
    def __init__(self, args):
        super(SIGMAModel, self).__init__(args)
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is not installed or failed to import. "
                f"Original error: {_MAMBA_IMPORT_ERR}. "
                "Install with: pip install mamba-ssm causal-conv1d"
            )

        self.args = args
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.sigma_blocks = nn.ModuleList([
            SIGMABlock(
                d_model=args.hidden_size,
                d_state=args.d_state,
                d_conv=args.d_conv,
                expand=args.expand,
                dropout=args.hidden_dropout_prob,
                num_layers=args.num_hidden_layers,
                max_seq_length=args.max_seq_length,
            )
            for _ in range(args.num_hidden_layers)
        ])

        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        x = self.item_embeddings(input_ids)
        x = self.dropout(x)
        x = self.LayerNorm(x)

        outputs = []
        for block in self.sigma_blocks:
            x = block(x)
            outputs.append(x)

        if all_sequence_output:
            return outputs
        return outputs[-1]

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_out = self.forward(input_ids)
        seq_out = seq_out[:, -1, :]
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return nn.CrossEntropyLoss()(logits, answers)
