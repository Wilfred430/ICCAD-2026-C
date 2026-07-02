"""Autoregressive B*-tree generator.

This is the "generative topology" model from WINNING_STRATEGY.md Stage 0:
instead of regressing (cx, cy, w, h) per block (which mode-collapses when a
case has several equally-good topologies -- averaging two valid layouts
produces an invalid one), this model learns to *construct* a B*-tree the
same way the 1M-case training set's `tree_sol` does: one block at a time,
each new block attached to an already-placed block via a (parent, direction)
pair.  Sampling from this model at inference time gives K different valid
topologies to hand to the packer, rather than one blurred-out average.

Architecture
------------

    block features [N, 16]           terminals [T, 2]
            |                               |
       (shared encoder, same design as model.py's FloorplanTransformer)
            |                               |
            +------------- context ---------+
                          |
              per-block embeddings h_i  [B, N, D]   (indexed by ORIGINAL
                          |                           block id, order-free)
        gather into generation order (gen_order[t] -> h_i)
                          |
                          v
              causal Transformer decoder over steps 0..N-1
              (step t may only attend to steps 0..t)
                          |
              per-step contextual features d_t  [B, N, D]
                          |
    +----------------+---------------+----------------+
    v                                v                v
block-selection pointer      parent-pointer         direction head
(query = d_{t-1} (shifted),  (query = d_t,          (MLP: d_t -> logit)
 keys  = h_i for i NOT        keys = d_0..d_{t-1},
 yet placed)                   causal-masked)

Three autoregressive decisions per step, all pointer networks (Vinyals et
al. 2015) rather than fixed-size classifiers, so the model works for any N:

  1. "Which block goes next?"      -- points into the REMAINING block set.
  2. "Which earlier step is its    -- points into the ALREADY-PLACED steps.
      parent?"
  3. "Which side does it attach    -- binary (left/right of parent).
      on?"

Decision 1 is what makes the model runnable standalone at inference: at
training time we teacher-force it with tree_sol's own DFS order, but at
inference (a brand new case with no tree_sol) the model must decide its own
generation order block-by-block -- see `generate()`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import BLOCK_FEAT_DIM


class BlockContextEncoder(nn.Module):
    """Same design as model.py's FloorplanTransformer encoder half: project
    block features + terminals into a shared hidden space and run full
    self-attention so every block embedding is aware of the whole case
    (connectivity, terminals, other blocks) before the tree is generated."""

    def __init__(self, block_feat_dim: int, hidden_dim: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.block_encoder = nn.Sequential(
            nn.Linear(block_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.term_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block_type_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.term_type_emb  = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.block_type_emb, std=0.02)
        nn.init.normal_(self.term_type_emb,  std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, blocks_feat, blocks_mask, terms, terms_mask):
        h_blocks = self.block_encoder(blocks_feat) + self.block_type_emb
        h_terms  = self.term_encoder(terms)        + self.term_type_emb
        h = torch.cat([h_blocks, h_terms], dim=1)
        full_mask = torch.cat([blocks_mask, terms_mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=~full_mask)
        n_blocks = blocks_feat.shape[1]
        return h[:, :n_blocks]  # [B, N, D], drop terminal slots


class TreeGenerator(nn.Module):
    def __init__(
        self,
        block_feat_dim: int = BLOCK_FEAT_DIM,
        hidden_dim: int = 128,
        n_ctx_layers: int = 4,
        n_dec_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_blocks: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_blocks = max_blocks

        self.context = BlockContextEncoder(block_feat_dim, hidden_dim, n_ctx_layers, n_heads, dropout)

        # Step (generation-order position) embedding -- the causal decoder
        # otherwise has no way to tell "this is step 5" from "this is step 50";
        # attention alone is permutation-equivariant.
        self.step_pos_emb = nn.Parameter(torch.zeros(1, max_blocks, hidden_dim))
        nn.init.normal_(self.step_pos_emb, std=0.02)

        # Learned "start" query used to predict the ROOT (step 0), i.e. the
        # block-selection decision made *before* anything has been placed.
        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.start_token, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_dec_layers)

        # Pointer networks (separate QK projections per decision so they
        # don't compete with the causal self-attention's own QKV):
        #   block-selection: query = state BEFORE step t, key = block ctx h_i
        #   parent-pointer:  query = state AT step t,     key = decoder d_j
        self.blocksel_query = nn.Linear(hidden_dim, hidden_dim)
        self.blocksel_key   = nn.Linear(hidden_dim, hidden_dim)
        self.ptr_query = nn.Linear(hidden_dim, hidden_dim)
        self.ptr_key   = nn.Linear(hidden_dim, hidden_dim)

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, blocks_feat, blocks_mask, terms, terms_mask, gen_order, step_mask):
        """Teacher-forced training pass -- computes all three heads for every
        step in parallel using the GROUND TRUTH gen_order as decoder input.

        gen_order: [B, N] int64, gen_order[b, t] = block id placed at step t
                   (padding rows can be anything valid; excluded via step_mask).
        step_mask: [B, N] bool, True for real (non-padding) generation steps.

        Returns:
            block_logits: [B, N, N] -- block_logits[b, t, i] = score of
                          block i being chosen at step t (i already placed
                          before t, or padding, is masked to -inf).
            ptr_logits:   [B, N, N] -- ptr_logits[b, t, j] = score of step j
                          being step t's parent (j >= t masked to -inf).
            dir_logits:   [B, N] -- direction bit logit for step t.
        """
        B, N = gen_order.shape
        h = self.context(blocks_feat, blocks_mask, terms, terms_mask)  # [B, N, D]

        gen_order_safe = gen_order.clamp(min=0)
        idx = gen_order_safe.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        seq = torch.gather(h, dim=1, index=idx)          # [B, N, D] in gen order
        seq = seq + self.step_pos_emb[:, :N]

        causal_mask = torch.triu(torch.ones(N, N, device=seq.device, dtype=torch.bool), diagonal=1)
        d = self.decoder(seq, mask=causal_mask, src_key_padding_mask=~step_mask)  # [B, N, D]

        # ---- 1. block selection: predict gen_order[t] from the state
        #         BEFORE step t (start_token for t=0, d[t-1] otherwise) ----
        query_state = torch.cat([self.start_token.expand(B, 1, self.hidden_dim), d[:, :-1]], dim=1)
        bq = self.blocksel_query(query_state)   # [B, N, D]
        bk = self.blocksel_key(h)               # [B, N, D] over ORIGINAL block ids
        block_logits = torch.matmul(bq, bk.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # [B, N, N]
        onehot = F.one_hot(gen_order_safe, num_classes=N).float()      # [B, N, N]
        used_before = onehot.cumsum(dim=1) - onehot                    # exclusive prefix
        invalid = (used_before > 0.5) | (~blocks_mask).unsqueeze(1)
        block_logits = block_logits.masked_fill(invalid, float("-inf"))

        # ---- 2. parent pointer: step t points at an earlier step j < t ----
        q = self.ptr_query(d)   # [B, N, D]
        k = self.ptr_key(d)     # [B, N, D]
        ptr_logits = torch.matmul(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # [B, N, N]
        block_future = torch.triu(torch.ones(N, N, device=seq.device, dtype=torch.bool), diagonal=0)
        ptr_logits = ptr_logits.masked_fill(block_future.unsqueeze(0), float("-inf"))

        # ---- 3. direction ----
        dir_logits = self.direction_head(d).squeeze(-1)  # [B, N]
        return block_logits, ptr_logits, dir_logits

    @torch.no_grad()
    def generate(self, blocks_feat, blocks_mask, terms, terms_mask, n_blocks: int,
                 temperature: float = 1.0, sample: bool = True, generator=None):
        """Autoregressive sampling for a SINGLE case (batch size 1), no
        ground-truth tree_sol required -- this is what runs at real
        inference time on unseen (validation/test) cases.

        Returns a dict with:
            gen_order:   [n_blocks] int64  -- block id placed at each step
            parent_id:   [n_blocks] int64  -- ORIGINAL block id of the
                         parent (-1 for root)
            direction:   [n_blocks] int64  -- 0/1 attach side (0 for root)
        """
        assert blocks_feat.shape[0] == 1, "generate() is single-case; batch the calls instead"
        device = blocks_feat.device
        D = self.hidden_dim
        h = self.context(blocks_feat, blocks_mask, terms, terms_mask)  # [1, N, D]
        N_pad = h.shape[1]

        gen_order  = torch.full((1, n_blocks), 0, dtype=torch.int64, device=device)
        parent_step = torch.full((n_blocks,), -1, dtype=torch.int64)
        direction   = torch.zeros(n_blocks, dtype=torch.int64)
        used_mask  = torch.zeros(N_pad, dtype=torch.bool, device=device)  # True = already placed

        prev_state = self.start_token.squeeze(0)  # [1, D], state "before step t"

        for t in range(n_blocks):
            # ---- pick the next block ----
            bq = self.blocksel_query(prev_state)               # [1, D]
            bk = self.blocksel_key(h[0])                        # [N_pad, D]
            logits = (bq @ bk.T).squeeze(0) / (D ** 0.5)        # [N_pad]
            logits = logits.masked_fill(used_mask, float("-inf"))
            logits = logits.masked_fill(~blocks_mask[0], float("-inf"))
            if sample:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator).item()
            else:
                nxt = int(torch.argmax(logits).item())
            used_mask[nxt] = True
            gen_order[0, t] = nxt

            # ---- run the causal decoder over steps 0..t to get d_t ----
            idx = gen_order[:, :t + 1].unsqueeze(-1).expand(-1, -1, D)
            seq = torch.gather(h, dim=1, index=idx) + self.step_pos_emb[:, :t + 1]
            causal_mask = torch.triu(torch.ones(t + 1, t + 1, device=device, dtype=torch.bool), diagonal=1)
            d_all = self.decoder(seq, mask=causal_mask)  # [1, t+1, D]
            d_t = d_all[:, -1]                            # [1, D] state AFTER step t

            if t == 0:
                parent_step[0] = -1
                direction[0] = 0
            else:
                q = self.ptr_query(d_t)                          # [1, D]
                k = self.ptr_key(d_all[0, :t])                    # [t, D]  (steps 0..t-1)
                p_logits = (q @ k.T).squeeze(0) / (D ** 0.5)      # [t]
                if sample:
                    p_probs = F.softmax(p_logits / max(temperature, 1e-6), dim=-1)
                    pstep = torch.multinomial(p_probs, 1, generator=generator).item()
                else:
                    pstep = int(torch.argmax(p_logits).item())
                parent_step[t] = pstep

                dir_logit = self.direction_head(d_t).squeeze()
                dir_prob = torch.sigmoid(dir_logit)
                if sample:
                    direction[t] = torch.bernoulli(dir_prob, generator=generator).long().item()
                else:
                    direction[t] = int((dir_prob > 0.5).item())

            prev_state = d_t

        parent_id = torch.full((n_blocks,), -1, dtype=torch.int64)
        for t in range(1, n_blocks):
            parent_id[t] = gen_order[0, parent_step[t]]

        return {
            "gen_order": gen_order[0].cpu(),
            "parent_id": parent_id,
            "direction": direction,
        }
