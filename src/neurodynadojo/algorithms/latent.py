"""Latent-embedding contenders — a third method family beside classical FC and system-ID.

  cebra   CEBRA (Schneider, Lee & Mathis 2023) contrastive latent embedding. Trained
          SELF-SUPERVISED (CEBRA-Time, no labels) on the pooled recordings, then each recording is
          embedded and summarised (mean + std over time) into a per-recording feature. This is the
          fair analogue of how a foundation model is probed: an unsupervised representation followed
          by the same cross-validated linear probe as every other contender — the label never
          touches the embedding.

Contract matches the rest of the zoo: `(N, C, T) -> (N, D)`. Needs the `[latent]` extra (cebra).
"""
from __future__ import annotations

import numpy as np


def cebra_features(X, dim=8, iters=300, device="cpu"):
    """One self-supervised CEBRA-Time model over all recordings; per-recording embedding pooled to
    mean+std over time. CPU by default (the pinned torch's cuDNN mismatches on this box's GPU)."""
    import cebra
    import torch
    torch.backends.cudnn.enabled = False
    Z = [np.ascontiguousarray(x.T.astype("float32")) for x in X]        # each (T, C)
    Z = [(z - z.mean(0)) / (z.std(0) + 1e-8) for z in Z]               # per-channel z-score
    model = cebra.CEBRA(model_architecture="offset10-model", batch_size=256,
                        output_dimension=dim, max_iterations=iters, temperature=1.0,
                        verbose=False, device=device)
    model.fit(np.concatenate(Z, axis=0))                               # CEBRA-Time, unsupervised
    feats = []
    for z in Z:
        e = model.transform(z)                                         # (T, dim)
        feats.append(np.concatenate([e.mean(0), e.std(0)]))
    return np.asarray(feats)


LATENT = {"CEBRA": cebra_features}
