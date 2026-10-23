# Physics-aware ablation study

The completed ablation study tested whether CSI augmentation should preserve the physical meaning of the radio-frequency axes rather than inherit photographic image augmentations unchanged.

## Experimental ladder

| Config | Change | Rank-1 | Rank-5 | mAP | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| M0 | Controlled baseline | 0.791 | 0.945 | 0.765 | 0.769 |
| M1 | Refreshed pair sampling | 0.883 | 0.963 | 0.843 | 0.867 |
| M2 | VisionAug control | 0.564 | 0.802 | 0.514 | 0.536 |
| M3 | SubCarrierAug | 0.885 | 0.977 | 0.857 | 0.878 |
| M4 | Dual-axis attention | **0.907** | 0.977 | **0.863** | **0.901** |
| M5 | Hard negatives / reported full system | 0.903 | 0.977 | 0.862 | 0.895 |

Each configuration was evaluated across three random seeds with bootstrap 95% confidence intervals.

## Main result

The controlled baseline rose from 0.791 Rank-1 to 0.903 for the designated full system, an increase of 11.1 percentage points. The best observed Rank-1 configuration was M4 at 0.907.

The strongest experimental contrast is M2 versus M3: generic image-style augmentation reached 0.564 Rank-1, while physics-grounded SubCarrierAug reached 0.885 under the matched stage of the ablation ladder. This supports the core project hypothesis that transformations which alter subcarrier ordering or scale can damage the discriminative radio-frequency structure.

## Interpretation of hard-negative mining

The completed table does **not** show an improvement from M4 to M5: Rank-1 moves from 0.907 to 0.903 and mAP from 0.863 to 0.862. The later refinement work attempted to redesign the mining procedure using semi-hard negatives; that follow-up is treated as exploratory rather than as a completed result.
