# Rover-Route

A Mars rover that **sees sand before it gets stuck**. A terrain model trained on real NASA
Curiosity images labels the ground ahead, and the rover plans around likely sand. We measure
how much safer that is than a rover that only checks slope.

## Why
Spirit got stuck in a sand trap in 2009 and never drove again. Slope alone can't tell soft sand
from solid ground; both can be flat. The rover has to look.

## How it works
1. **Real terrain data.** 149 top-down tiles built from AI4Mars (Curiosity NAVCAM images with
   human labels: soil, bedrock, sand, big rock). Each tile is 20 x 20 cells. `data/preds20.npz`
   holds a ResNet34 U-Net's calibrated class probabilities for every cell (test mIoU 0.80,
   temperature 1.52); `data/tiles.npz` holds the true labels.
2. **World.** Each seed stitches 3 x 3 tiles into a 60 x 60 grid (0.5 m cells, 30 m across) and
   adds a random slope field. The true terrain is hidden from the rover.
3. **Perception.** The rover sees 3 m around itself. It always senses slope; depending on the
   mode it also gets the model's prediction (Rover-Route) or the true class (oracle).
4. **Cost map.** `cost = 1 + slope + terrain risk + uncertainty`. Too steep, likely sand
   (P >= 0.5) and likely big rock are blocked.
5. **Planning.** 8-connected A*, replanning every 4 steps or as soon as the route ahead is blocked.
6. **Evaluation.** Every mode drives the same seeded worlds. Hazard entries are scored against
   the true terrain.

## Results
100 seeded worlds, every rover on the same worlds (`python -m scripts.run_eval`, full table in
[`results/results.md`](results/results.md)):

| Rover | Success [95% CI] | Reached goal | Hazard entries / mission | Route ratio |
| --- | --- | --- | --- | --- |
| Geometry only (slope) | 0.46 [0.36, 0.55] | 1.00 | 0.71 | 1.009 |
| **Rover-Route (model)** | **0.72** [0.63, 0.80] | 0.76 | **0.06** | 1.086 |
| Oracle (true terrain) | 1.00 | 1.00 | 0.00 | 1.080 |

- Rover-Route enters hazards **about 12x less often** than the slope-only rover, for routes
  about 8% longer than the safest possible route.
- Its main failure is being too careful. In 24 of 100 worlds, predicted sand closed every route
  and it stopped safely instead of reaching the goal.

## Run it
```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py                 # mission view (demo seed 18)
python -m scripts.run_eval --n 100   # regenerate results/
python -m scripts.pick_demo          # seeds where the model visibly detours
python -m pytest -q
```

## Layout
```
app.py                 Streamlit mission view
roverroute/            data, world, perception, costmap, planner, rover, metrics, card
scripts/               run_eval.py, pick_demo.py
data/                  tile labels, model predictions, camera frames + overlays
results/               measured results
tests/                 planner, metrics, rover, decision card
```

## Honesty notes
- The images come from Curiosity at Gale crater. The map is a synthetic grid stitched from
  unrelated images, so tile edges don't line up, and the slope field is random.
- The tiles come from a flat-ground projection without camera calibration.
- The tile library has almost no big rocks, so the hazards here are sand and steep slope.
- Model predictions are precomputed per cell; the app does not run the network live.
- Rover motion and sensing are simplified. This is a proof of the idea, not flight software.
