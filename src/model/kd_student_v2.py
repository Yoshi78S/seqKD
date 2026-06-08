"""KDStudent v2 — GLINT-RU-inspired lightweight student.

Per-block architecture (GLINT-RU style):

    x  ─► Linear ─► CausalConv1D ─► GRU ─► SelectiveGate ─► Dropout ─► LN
                                                                       │
                                                                       ▼
                                                                    GatedMLP

Where:
    SelectiveGate(conv_out, gru_out) = Linear_g2(SiLU(Linear_g1(conv_out)))
                                       ⊙ Linear_v(gru_out)
    GatedMLP(z)                      = LN(z + Dropout(Linear_o(
                                            GeLU(Linear_g(z)) ⊙ Linear_v(z))))

Ablation flags (shared with v1 where meaningful, plus three v2-only):
    abl_no_pos_emb     : drop position embeddings
    abl_no_input_ln    : drop input LayerNorm + Dropout
    abl_no_block_ln    : drop block-internal LayerNorm (after gate)
    abl_no_conv        : drop Linear→CausalConv1D in front of GRU
    abl_no_gate        : drop SelectiveGate (raw GRU output downstream)
    abl_no_gated_mlp   : drop GatedMLP (block ends after gate+LN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm


# ─── Causal Conv1D (left-pad, no future leak) ────────────────────

class CausalConv1d(nn.Module):
    """Conv1D over the time axis with strict causal padding.

    Input shape : (batch, seq_len, channels)
    Output shape: (batch, seq_len, channels)
    The output at position t depends only on inputs 0..t.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # No nn.Conv1d padding — we pad manually on the left only.
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                              padding=0)

    def forward(self, x):
        # x: (B, L, C) → (B, C, L)
        x = x.transpose(1, 2)
        # Left-pad with (kernel_size - 1) zeros, right-pad with 0.
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)  # (B, C, L)
        return x.transpose(1, 2)  # → (B, L, C)


# ─── Selective Gate (GLINT-RU Eq. 9) ─────────────────────────────

class SelectiveGate(nn.Module):
    """SiLU(Linear1(C)) → Linear2 → ⊙ Linear3(H).

    Args:
        d : hidden size (same on both gate and value paths)
    Forward inputs:
        conv_out : (B, L, d) — the conv stream (GRU input)
        gru_out  : (B, L, d) — the GRU output

    The forward output is what HS-KD should align with.
    """
    def __init__(self, d):
        super().__init__()
        self.gate_in = nn.Linear(d, d)     # Linear1
        self.gate_out = nn.Linear(d, d)    # Linear2 (after SiLU)
        self.value = nn.Linear(d, d)       # Linear3 (channel mix on GRU out)

    def forward(self, conv_out, gru_out):
        gate = self.gate_out(F.silu(self.gate_in(conv_out)))
        value = self.value(gru_out)
        return gate * value


# ─── Gated MLP (GLINT-RU Eq. 12) ─────────────────────────────────

class GatedMLP(nn.Module):
    """GeLU(Linear_g(z)) ⊙ Linear_v(z) → Linear_o → Dropout → +residual → LN.

    Replaces the heavier FeedForward from v1 (~33K → ~12K params / block).
    """
    def __init__(self, d, dropout):
        super().__init__()
        self.gate = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(d, eps=1e-12)

    def forward(self, z):
        gated = F.gelu(self.gate(z)) * self.value(z)
        h = self.dropout(self.out(gated))
        return self.layernorm(h + z)


# ─── Student block ────────────────────────────────────────────────

class StudentBlockV2(nn.Module):
    def __init__(self, args):
        super().__init__()
        d = args.hidden_size
        self.use_conv = not getattr(args, 'abl_no_conv', False)
        self.use_gate = not getattr(args, 'abl_no_gate', False)
        self.use_gated_mlp = not getattr(args, 'abl_no_gated_mlp', False)
        self.use_block_ln = not getattr(args, 'abl_no_block_ln', False)

        # GRU front-end: optional Linear → CausalConv1D
        if self.use_conv:
            k = getattr(args, 'conv_kernel_size', 3)
            self.pre_conv_linear = nn.Linear(d, d)
            self.causal_conv = CausalConv1d(d, kernel_size=k)

        self.gru = nn.GRU(
            input_size=d, hidden_size=d, num_layers=1,
            batch_first=True, bias=False,
        )

        if self.use_gate:
            self.selective_gate = SelectiveGate(d)
            # HS-KD aligns the post-gate sequence with the teacher's
            # pre-residual attention output.
            self.hs_hook_target = self.selective_gate
        else:
            self.hs_hook_target = self.gru

        self.gru_dropout = nn.Dropout(args.hidden_dropout_prob)
        if self.use_block_ln:
            self.gru_layernorm = LayerNorm(d, eps=1e-12)

        if self.use_gated_mlp:
            self.gated_mlp = GatedMLP(d, dropout=args.hidden_dropout_prob)
        else:
            # No transform; block ends with the gate's output (or raw GRU
            # if abl_no_gate too). Ablating GatedMLP also drops its
            # residual + LN, matching how v1's abl_no_ffn worked.
            self.gated_mlp = None

    def forward(self, hidden_states):
        # GRU input (optionally Linear → CausalConv)
        if self.use_conv:
            conv_out = self.causal_conv(self.pre_conv_linear(hidden_states))
        else:
            conv_out = hidden_states

        self.gru.flatten_parameters()
        gru_out, _ = self.gru(conv_out)

        # Selective gate (or pass-through)
        if self.use_gate:
            gru_out = self.selective_gate(conv_out, gru_out)

        gru_out = self.gru_dropout(gru_out)
        if self.use_block_ln:
            gru_out = self.gru_layernorm(gru_out)

        if self.gated_mlp is not None:
            return self.gated_mlp(gru_out)
        return gru_out


# ─── Encoder ─────────────────────────────────────────────────────

class StudentEncoderV2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.blocks = nn.ModuleList([
            StudentBlockV2(args) for _ in range(args.num_hidden_layers)
        ])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


# ─── Model ───────────────────────────────────────────────────────

class KDStudentV2Model(SequentialRecModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.no_pos_emb = getattr(args, 'abl_no_pos_emb', False)
        self.no_input_ln = getattr(args, 'abl_no_input_ln', False)
        # LayerNorm / Dropout for input stream; instantiated even when the
        # flag turns them off so init_weights() finds them, but unused in
        # the forward pass.
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.student_encoder = StudentEncoderV2(args)
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
        # Dropout is always applied per the v2 design — see report comment.
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self._embed(input_ids)
        return self.student_encoder(sequence_emb)

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)[:, -1, :]
        logits = torch.matmul(seq_output, self.item_embeddings.weight.T)
        return nn.CrossEntropyLoss()(logits, answers)

    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        return self.forward(input_ids)
