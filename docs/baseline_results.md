# Baseline study

The first completed experiment established a compact CSI-amplitude baseline before introducing later training and architectural changes.

## Setup

- Dataset: Wi-PER81, 81 identities, two acquisition sessions per identity.
- Representation: numeric CSI amplitude windows rather than rendered RGB heatmaps.
- Window shape: `1 x 100 x 52`.
- Architecture: two convolution/pooling blocks, subcarrier attention, 128-dimensional signature embedding, identity-classification auxiliary head.
- Evaluation: closed-set session-B probe/gallery retrieval.

## Initial result

| Metric | Value |
| --- | ---: |
| Rank-1 | 0.778 |
| Rank-5 | 0.909 |
| mAP | 0.752 |
| Macro-F1 | 0.756 |
| Raw-cosine no-learning floor (Rank-1) | 0.342 |

The run demonstrated learned cross-session identity structure, but the training curves showed severe early overfitting.

## Failure diagnosis

The key issue was not just model capacity: the training pair set was constructed once before the epoch loop and then repeatedly reshuffled. With roughly 6,000 fixed pairs and a ~10.7M-parameter network, the model could memorize the finite pair set within a few epochs.

That diagnosis motivated epoch-refreshed pair sampling in the next experiment series.
