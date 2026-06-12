"""KDStudent v4: v3 (FreqMamba) + bipolar recency gate (rank-1 modulation).

The v3 body is UNCHANGED (subclass). At the final position, the readout state
h is modulated along the last-item embedding direction:

    g  = tanh(MLP(f))           f = 8 standardized, DETACHED gate features
    h~ = h + g * (h . e_l) * e_l        e_l = E_s[ell]/||E_s[ell]||

g = -1 removes the recency component (the G2' minus-side oracle), g > 0 adds
it, g = 0 is exactly v3. predict() returns the sequence output with the last
position replaced by h~, so every existing readout (z~ = h~ @ E^T), eval and
leak path works unmodified. The gate is part of the model (eval uses it too).

Features (G3 Tier-B + length & popularity; all no_grad, standardized with
buffers measured once at init calibration):
    [cos(h,e_l), (h.e_l)/||h||, log||h||, top1-top2 margin(z),
     entropy(softmax(z)), percentile_rank(z_ell), log(real len), log1p(pop[ell])]
NO privileged quantity (kappa / a / y) enters the gate input.

gate_fix: None (learned) | -1.0 | 0.0  (fixed-gate control arms).
Final gate layer is zero-initialized -> g == 0 at step 0 (= v3 starting point).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kd_student_v3 import KDStudentV3Model


class KDStudentV4Model(KDStudentV3Model):
    def __init__(self, args=None):
        super().__init__(args=args)
        d = args.hidden_size
        self.gate_fc1 = nn.Linear(8, 16)
        self.gate_fc2 = nn.Linear(16, 1)
        nn.init.zeros_(self.gate_fc2.weight)            # g == 0 at start (S5)
        nn.init.zeros_(self.gate_fc2.bias)
        self.register_buffer("feat_mu", torch.zeros(8))
        self.register_buffer("feat_sd", torch.ones(8))
        self.register_buffer("pop_log", torch.zeros(args.item_size))
        self.gate_fix = None                            # None | -1.0 | 0.0
        self.last_g = None                              # set on every predict

    @torch.no_grad()
    def gate_features(self, h, input_ids):
        """8 raw (unstandardized) features; caller standardizes."""
        E = self.item_embeddings.weight
        ell = input_ids[:, -1]
        e_l = F.normalize(E[ell], dim=-1)
        hn = h.norm(dim=-1).clamp_min(1e-9)
        p = (h * e_l).sum(-1)
        z = h @ E.T
        top2 = z.topk(2, dim=1).values
        pz = F.softmax(z, -1)
        ent = -(pz * pz.clamp_min(1e-9).log()).sum(-1)
        prank = (z > z.gather(1, ell[:, None])).float().mean(1)
        ln = (input_ids != 0).sum(1).float().clamp_min(1)
        return torch.stack([p / hn, p / hn, hn.log(),
                            top2[:, 0] - top2[:, 1], ent, prank,
                            ln.log(), self.pop_log[ell]], dim=1)

    def gate(self, h, input_ids):
        if self.gate_fix is not None:
            g = torch.full((h.size(0),), float(self.gate_fix),
                           device=h.device, dtype=h.dtype)
            self.last_g = g
            return g
        f = self.gate_features(h, input_ids)            # no_grad (S2)
        f = (f - self.feat_mu) / self.feat_sd
        g = torch.tanh(self.gate_fc2(F.relu(self.gate_fc1(f)))).squeeze(-1)
        self.last_g = g
        return g

    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        seq = super().predict(input_ids, user_ids)
        h = seq[:, -1, :]
        g = self.gate(h, input_ids)
        E = self.item_embeddings.weight
        e_l = F.normalize(E[input_ids[:, -1]], dim=-1)
        h_t = h + (g * (h * e_l).sum(-1)).unsqueeze(-1) * e_l
        return torch.cat([seq[:, :-1, :], h_t.unsqueeze(1)], dim=1)
