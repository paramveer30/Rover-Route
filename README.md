# Rover-Route

A simulated Mars rover that uses a terrain model to avoid sand.

The Spirit rover got stuck in sand in 2009 and never moved again. Sand is usually flat, so a rover that only checks slope can't tell it apart from solid ground. This project gives the rover a model that looks at the ground and guesses what it is (soil, bedrock, sand or rock), so it can plan around sand.

## How it works

The terrain model was trained on real Curiosity camera images from the AI4Mars dataset. The training code is in the ml folder. Its predictions are saved ahead of time, so the app itself doesn't need PyTorch.

Each world is built from real labelled tiles stitched together, with some random hills added for slope. The rover can only see a few metres around itself. Each step it looks around, builds a cost map, plans a route with A*, and moves.

There are three rovers to compare:

- one that only checks slope
- one that also uses the terrain model (Rover-Route)
- one that is given the real terrain for the cells it can see, as a best case

The model rover hits hazards far less often than the slope-only one. Sometimes it's too careful and stops when sand seems to block every path.

## Running it

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To run the full evaluation: `python -m scripts.run_eval`

To run the tests: `python -m pytest`

## Retraining the model

You only need this if you want to train the model yourself. Install the extra packages with `pip install -r requirements-ml.txt`, then run `python -m ml.train`. The weights aren't in git because they're too big.

## Notes

The maps are stitched from unrelated photos, so they aren't a real location on Mars. This is a proof of concept, not real rover software.
