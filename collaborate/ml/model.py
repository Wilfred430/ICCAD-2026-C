"""Graph-Transformer for floorplan position prediction.

Architecture summary
--------------------

    block features [N, 16]                terminals [T, 2]
            |                                    |
       block_encoder (Linear -> ReLU -> Linear)  term_encoder (Linear)
            |                                    |
            +---------- concatenate -------------+
                              |
                              v
                ┌─────────────────────────────┐
                │  TransformerEncoder × L     │   self-attention over
                │  (multi-head, batch_first)  │   {blocks ∪ terminals}
                └─────────────────────────────┘
                              |
                              v
            split blocks back out (drop terminal slots)
                              |
              ┌───────────────┴────────────────┐
              v                                v
        pos_head (Linear → 2)            dim_head (Linear → 2)
            (cx, cy)                         (w,  h)

Cross-attention between blocks and terminals is implicit: by feeding
terminal embeddings as additional sequence positions, every transformer
layer attends bidirectionally.  This gives the model a way to learn
"block i strongly connected to pins in region X → block i should sit
near X".

Why a Transformer and not a GCN?  The b2b/p2b nets are sparse but the
real coupling is "global" — block A might never share a net with block B,
but they both pull on the same terminal and so must coordinate.  Vanilla
GCN message-passing has trouble with that long-range coordination at
small depth; full self-attention handles it for free.

Number of params (defaults): ~250K. Fits on CPU comfortably; trains in
30-60 min on a modern GPU for ~5 epochs over 10K cases.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .data import BLOCK_FEAT_DIM


class FloorplanTransformer(nn.Module):
    def __init__(
        self,
        block_feat_dim: int = BLOCK_FEAT_DIM,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Encoders project block & terminal raw features into hidden_dim.
        self.block_encoder = nn.Sequential(
            nn.Linear(block_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Terminals are just (x, y) -> need a learnable projection plus a
        # type embedding so the model knows "this is a terminal, not a block".
        self.term_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block_type_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.term_type_emb  = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Stack of Transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: more stable for deep transformers
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoders: predict (cx, cy) and (w, h) per block.
        self.pos_head = nn.Linear(hidden_dim, 2)
        self.dim_head = nn.Linear(hidden_dim, 2)

        # Initialise type embeddings small so the model starts ~symmetric.
        nn.init.normal_(self.block_type_emb, std=0.02)
        nn.init.normal_(self.term_type_emb,  std=0.02)

    def forward(
        self,
        blocks_feat: torch.Tensor,   # [B, N, F]
        blocks_mask: torch.Tensor,   # [B, N]    True = real block
        terms:       torch.Tensor,   # [B, T, 2]
        terms_mask:  torch.Tensor,   # [B, T]    True = real terminal
    ):
        # ---- encode ---------
        h_blocks = self.block_encoder(blocks_feat) + self.block_type_emb
        h_terms  = self.term_encoder(terms)        + self.term_type_emb

        # ---- combine block + terminal sequence ----
        h = torch.cat([h_blocks, h_terms], dim=1)            # [B, N+T, D]
        full_mask  = torch.cat([blocks_mask, terms_mask], 1) # True = valid
        # TransformerEncoder expects key_padding_mask = True for padding.
        key_padding_mask = ~full_mask

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        # ---- decode block positions only -----
        n_blocks = blocks_feat.shape[1]
        h_blocks_out = h[:, :n_blocks]
        pos = self.pos_head(h_blocks_out)   # [B, N, 2]  -> (cx, cy)
        dim = self.dim_head(h_blocks_out)   # [B, N, 2]  -> (w,  h)

        # Make w/h strictly positive via softplus (so we don't predict
        # negative dimensions which the packer would reject).
        dim = nn.functional.softplus(dim) + 1e-3
        return pos, dim
