"""ML floorplanner for ICCAD 2026 FloorSet Challenge.

Two model families live here:

  1. Coordinate regression (original): a graph-Transformer predicts
     (cx, cy, w, h) per block directly.  Prone to mode collapse on cases
     with multiple valid topologies (averaging two valid layouts produces
     an overlapping, invalid one) -- see WINNING_STRATEGY.md Section 1.
  2. Generative B*-tree (current focus): an autoregressive model
     *constructs* a B*-tree one block at a time (which block goes next,
     which earlier block it attaches to, which side), trained on the 1M
     training set's own `tree_sol` labels.  Sampling K topologies and
     packing each one avoids the mode-collapse failure entirely.

Modules:
    data         -- FloorSet Lite dataset loader; also derives B*-tree
                    generation-order targets from `tree_sol` (TRAIN format)
    model        -- Transformer-based coordinate regressor (family 1)
    train        -- Supervised training loop for model.py (family 1)
    predict      -- Inference helper used by my_optimizer_ml.py at solve-time
    model_tree   -- TreeGenerator: autoregressive B*-tree constructor (family 2)
    train_tree   -- Teacher-forced training loop for model_tree.py
    pack_tree    -- Python port of src/packer.cpp's contour packer, for
                    turning a sampled topology into (x, y, w, h) geometry
    run_pipeline -- One-command demo: train (if needed) -> sample -> pack ->
                    score -> save; `python -m ml.run_pipeline`
"""
