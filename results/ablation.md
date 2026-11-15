# Ablation: model choices vs rover outcome

Model rover only, seeds 0..99, same worlds for every row (`python -m scripts.ablation`).

| Config | Success [95% CI] | Reached goal | Hazard entries / mission | Route ratio |
| --- | --- | --- | --- | --- |
| calibrated, block P>=0.5 (shipped) | 0.72 [0.63, 0.80] | 0.76 | 0.06 | 1.086 |
| raw T=1, block P>=0.5 | 0.71 [0.62, 0.79] | 0.76 | 0.07 | 1.076 |
| calibrated, block P>=0.3 | 0.72 [0.63, 0.80] | 0.74 | 0.04 | 1.073 |
| raw T=1, block P>=0.3 | 0.72 [0.63, 0.80] | 0.75 | 0.05 | 1.063 |
| calibrated, block P>=0.7 | 0.76 [0.67, 0.84] | 0.80 | 0.06 | 1.085 |
| raw T=1, block P>=0.7 | 0.73 [0.64, 0.81] | 0.78 | 0.07 | 1.075 |
| calibrated, no uncertainty cost | 0.71 [0.62, 0.79] | 0.76 | 0.09 | 1.097 |
