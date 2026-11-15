# Results

100 seeded worlds (seeds 0..99), every mode on the same worlds. git `2e71e69`, model temperature 1.5187, 62.6 s.

| Rover | Success [95% CI] | Reached goal | Hazard entries / mission [95% CI] | Route ratio (reached) |
| --- | --- | --- | --- | --- |
| Geometry only (slope) | 0.46 [0.36, 0.55] | 1.00 | 0.71 [0.57, 0.86] | 1.009 (n=100) |
| Rover-Route (model) | 0.72 [0.63, 0.80] | 0.76 | 0.06 [0.01, 0.12] | 1.086 (n=76) |
| Oracle (true terrain) | 1.00 [1.00, 1.00] | 1.00 | 0.00 [0.00, 0.00] | 1.080 (n=100) |

## Definitions

- **Hazard**: true sand, true big rock, or slope over 20 degrees.
- **Hazard entry**: the rover drives onto hazard cells; a continuous run counts once.
- **Success**: reached the goal with 0 hazard entries.
- **Route ratio**: distance driven / shortest route that never touches a true hazard.
- **CI**: 95% percentile bootstrap of the mean.
