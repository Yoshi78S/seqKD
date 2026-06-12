"""KDStudent v3 — FreqMamba: BSARec's two-branch fusion with attention → Mamba.

This student *inherits BSARec's actual architecture* and replaces only the
O(n^2) self-attention branch with an O(n) residual-free Mamba branch. It is NOT
the serial "FFT-filter → Mamba" of EchoMamba4Rec; it is the **parallel
branch-fusion** that BSARec itself uses:

    BSARecLayer:   h = α · FrequencyLayer(x)  +  (1-α) · MultiHeadAttention(x)
                            └─ freq branch ─┘        └──── attention branch ────┘

    FreqMambaLayer (ours):
                   h = α · FrequencyLayer(x)  +  (1-α) · LN(Mamba(x))
                            └─ identical to ─┘        └── attention → Mamba, ───┘
                               BSARec's freq             NO fixed residual
                               branch (β, c)

Why this matches the residual-dominance story:
  * The frequency branch is *byte-for-byte BSARec's* `FrequencyLayer`
    (low/high split + learnable per-dim high-freq rescaling β = sqrt_beta²),
    so the teacher's frequency inductive bias is structurally inherited
    (and β / α can even be copied at init — see notes).
  * The attention branch — the source of residual dominance (h = x + Attn(x))
    — is replaced by Mamba with NO fixed residual:  ssp = LN(Mamba(x)).
    Mamba's input-dependent decay Ā_t ∈ (0,1)^N is an *adaptive* residual,
    not a fixed 1:1 one.
  * The fusion (α blend) and the per-block FFN are identical to BSARec, so the
    teacher↔student correspondence is branch-for-branch.

Distillation correspondence (handled by KDStudentDistillTrainer):
  teacher pre-residual attn output  `blocks[i].layer.attention_layer.dense`
  ↔ student Mamba branch output     `blocks[i].layer.mamba`  (= hs_hook_target)

Block-internal layout per layer ℓ:
  dsp = FrequencyLayer(x)                         (BSARec-identical)
  ssp = LN(Mamba(x))                              (NO fixed residual)
  h   = α · dsp + (1-α) · ssp                     (BSARec-identical fusion)
  x'  = LN(FFN(h) + h)                            (compressed FFN, inner = 2·d)

Ablation flags (all on `args`, default False):
  abl_no_pos_emb      : drop position embeddings
  abl_no_input_ln     : drop the input LayerNorm + dropout
  abl_no_ffn          : drop the per-block FFN (identity)
  abl_no_block_ln     : drop the LayerNorm on the Mamba branch (raw Mamba out)
  abl_no_freq         : drop the frequency branch (pure Mamba, α ignored)
  abl_mamba_residual  : re-add the fixed `+ x` residual on the Mamba branch
                        (tests the core residual-dominance claim)

Notes / available extensions (not enabled by default):
  * α (fusion weight) is a fixed hyperparameter `args.alpha`, matching BSARec.
    It can be made learnable or initialised from the teacher's α.
  * The frequency branch's β (sqrt_beta) and the fusion α can be copied from
    the teacher checkpoint at init as a third knowledge-transfer path
    (besides Pred-KD and HS-KD) — a natural ablation item.
"""
import copy
import torch
import torch.nn as nn

from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm
from model.bsarec import FrequencyLayer  # byte-for-byte BSARec frequency branch


class StudentFrequencyLayer(nn.Module):
    """BSARec's FrequencyLayer with an OPTIONAL ablation of its internal fixed
    residual:  LN(filtered + x)  ->  LN(filtered)   (--abl_no_freq_residual).

    Default (flag off) is byte-identical to bsarec.FrequencyLayer, and the
    attribute names (sqrt_beta / LayerNorm / out_dropout) match it exactly so
    state_dict keys are unchanged.
    Note: even with the residual removed, the filter itself passes an identity
    component beta^2 * x (filtered = beta^2*x + (1-beta^2)*low_pass), so this
    ablates the EXPLICIT residual only; the learned beta is reported after
    training to check for identity re-emergence (beta -> 1).
    """

    def __init__(self, args):
        super().__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.c = args.c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))
        self.use_residual = not getattr(args, 'abl_no_freq_residual', False)

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        if self.use_residual:
            return self.LayerNorm(hidden_states + input_tensor)
        return self.LayerNorm(hidden_states)

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
    _MAMBA_IMPORT_ERR = None
except Exception as _e:  # pragma: no cover - environment dependent
    Mamba = None
    _MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERR = _e


class CompressedFeedForward(nn.Module):
    """Position-wise FFN W2·GELU(W1·) with inner = 2·d (compressed from the
    teacher's 4·d), keeping the residual + LN. The FFN residual is retained
    because it does not mix along the sequence axis and is therefore not a
    source of residual dominance.
    """

    def __init__(self, args):
        super().__init__()
        d = args.hidden_size
        inner = 2 * d
        self.dense_1 = nn.Linear(d, inner)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(inner, d)
        self.LayerNorm = LayerNorm(d, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, x):
        h = self.dense_2(self.activation(self.dense_1(x)))
        h = self.dropout(h)
        return self.LayerNorm(h + x)


class FreqMambaLayer(nn.Module):
    """BSARecLayer with the attention branch replaced by a residual-free Mamba.

        dsp = FrequencyLayer(x)                       # freq branch (BSARec-identical)
        ssp = LayerNorm(Mamba(x))                     # Mamba branch (no fixed residual)
        out = alpha * dsp + (1 - alpha) * ssp         # fusion (BSARec-identical)
    """

    def __init__(self, args):
        super().__init__()
        self.use_freq = not getattr(args, 'abl_no_freq', False)
        self.use_block_ln = not getattr(args, 'abl_no_block_ln', False)
        self.mamba_residual = getattr(args, 'abl_mamba_residual', False)
        self.alpha = args.alpha

        if self.use_freq:
            self.filter_layer = StudentFrequencyLayer(args)

        self.mamba = Mamba(
            d_model=args.hidden_size,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
        )
        if self.use_block_ln:
            self.mamba_ln = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        # Mamba branch (replaces attention), no fixed residual by default.
        m = self.mamba(input_tensor)
        if self.mamba_residual:
            m = m + input_tensor
        ssp = self.mamba_ln(m) if self.use_block_ln else m

        if not self.use_freq:
            return ssp

        dsp = self.filter_layer(input_tensor)
        return self.alpha * dsp + (1 - self.alpha) * ssp


class FreqMambaBlock(nn.Module):
    """FreqMambaLayer + compressed FeedForward (inner = 2·d)."""

    def __init__(self, args):
        super().__init__()
        self.layer = FreqMambaLayer(args)
        self.use_ffn = not getattr(args, 'abl_no_ffn', False)
        if self.use_ffn:
            self.feed_forward = CompressedFeedForward(args)
        # HS-KD / CDD align the Mamba branch output with the teacher's
        # pre-residual attention output.
        self.hs_hook_target = self.layer.mamba

    def forward(self, hidden_states):
        layer_output = self.layer(hidden_states)
        if self.use_ffn:
            return self.feed_forward(layer_output)
        return layer_output


class FreqMambaEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.blocks = nn.ModuleList([
            FreqMambaBlock(args) for _ in range(args.num_hidden_layers)
        ])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class KDStudentV3Model(SequentialRecModel):
    def __init__(self, args):
        super().__init__(args)
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is not installed or failed to import. "
                f"Original error: {_MAMBA_IMPORT_ERR}. "
                "Install with: pip install mamba-ssm causal-conv1d")

        self.args = args
        self.no_pos_emb = getattr(args, 'abl_no_pos_emb', False)
        self.no_input_ln = getattr(args, 'abl_no_input_ln', False)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        # Real submodule named `student_encoder` (matches v1/v2 + the
        # KDStudentDistillTrainer contract). `item_encoder` is exposed as a
        # read-only alias below for BSARec-parallel hook paths.
        self.student_encoder = FreqMambaEncoder(args)

        # init_weights would clobber Mamba's carefully crafted internal init
        # (dt_proj.bias, A_log, D, special Linear inits). Snapshot every Mamba's
        # state, run init_weights on the whole model, then restore them.
        mamba_states = {
            name: copy.deepcopy(m.state_dict())
            for name, m in self.named_modules() if isinstance(m, Mamba)
        }
        self.apply(self.init_weights)
        for name, m in self.named_modules():
            if isinstance(m, Mamba):
                m.load_state_dict(mamba_states[name])

    @property
    def item_encoder(self):
        """Alias for BSARec-parallel hook paths (`item_encoder.blocks[i]...`).

        A property (not a submodule) so it does not duplicate parameters or
        state_dict keys.
        """
        return self.student_encoder

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
        return self.student_encoder(sequence_emb)

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)[:, -1, :]
        logits = torch.matmul(seq_output, self.item_embeddings.weight.T)
        return nn.CrossEntropyLoss()(logits, answers)

    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        return self.forward(input_ids)
