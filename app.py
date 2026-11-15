"""Rover-Route mission view.   streamlit run app.py"""
from __future__ import annotations

import time

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from roverroute.card import decision_card
from roverroute.data import frame_path, overlay_path
from roverroute.metrics import hazard_entries, score
from roverroute.rover import run_mission
from roverroute.world import MAX_SLOPE_DEG, SIZE, build_world, hazard_mask

SCALE = 9
COLOURS = np.array([[200, 161, 101], [138, 143, 152], [242, 209, 107], [229, 72, 77]], np.float32)
LABELS = {"geometry": "Geometry only (slope)", "model": "Rover-Route (model)", "oracle": "Oracle (true terrain)"}
DEMO_SEED = 18

st.set_page_config(page_title="Rover-Route", layout="wide")


@st.cache_resource
def simulate(seed: int):
    world = build_world(seed)
    missions = {m: run_mission(world, m, record=True) for m in LABELS}
    return world, missions


def render(world, mission, step: int, show_truth: bool) -> Image.Image:
    frame = mission.frames[min(step, len(mission.frames) - 1)]
    shade = 0.55 + 0.45 * (1 - np.clip(world.slope / MAX_SLOPE_DEG, 0, 1))
    rgb = np.repeat((90 * shade)[..., None], 3, axis=2)
    if mission.mode == "geometry":
        rgb[frame.seen] = (150 * shade[frame.seen])[:, None]
    else:
        cls = mission.belief.probs.argmax(axis=-1) if mission.mode == "oracle" else world.probs.argmax(axis=-1)
        rgb[frame.known] = COLOURS[cls[frame.known]] * shade[frame.known, None]
    steep = frame.seen & (world.slope > MAX_SLOPE_DEG)
    rgb[steep] = [70, 40, 90]
    if show_truth:
        hz = hazard_mask(world)
        rgb[hz] = 0.4 * rgb[hz] + 0.6 * np.array([255, 0, 0])
    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize((SIZE * SCALE,) * 2, Image.NEAREST)
    draw = ImageDraw.Draw(img)

    def xy(c):
        return (c[1] * SCALE + SCALE // 2, c[0] * SCALE + SCALE // 2)

    if len(frame.path) > 1:
        draw.line([xy(c) for c in frame.path], fill=(80, 220, 255), width=2)
    driven = mission.trajectory[: step + 1]
    if len(driven) > 1:
        draw.line([xy(c) for c in driven], fill=(255, 255, 255), width=3)
    for cell, colour in ((world.goal, (60, 230, 120)), (frame.cell, (255, 255, 255))):
        x, y = xy(cell)
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=colour, outline=(0, 0, 0), width=2)
    return img


def camera(world, cell) -> Image.Image:
    tile = world.tile_at(cell)
    base = Image.open(frame_path(tile)).convert("RGBA")
    return Image.alpha_composite(base, Image.open(overlay_path(tile)).convert("RGBA").resize(base.size))


st.title("Rover-Route")
st.caption("A Mars rover that uses a terrain model trained on real NASA Curiosity images to plan around sand.")

with st.sidebar:
    seed = st.number_input("World seed", 0, 999, DEMO_SEED)
    right = st.selectbox("Compare against", ["model", "oracle"], format_func=LABELS.get)
    show_truth = st.toggle("Show true hazards", False)
    st.markdown("**Legend**  \n🟫 soil · ⬜ bedrock · 🟨 sand · 🟥 big rock  \n🟪 too steep · "
                "dark = unseen  \ncyan = planned route · white = driven")

world, missions = simulate(int(seed))
pair = [missions["geometry"], missions[right]]
last = max(len(m.frames) for m in pair) - 1
step = st.slider("Step", 0, last, last, key=f"step-{seed}-{right}")
play = st.button("▶ Play mission")

hazards = hazard_mask(world)
cols = st.columns(2)
slots = [(c.empty(), c.empty()) for c in cols]


def draw(step: int) -> None:
    for (title, image), mission in zip(slots, pair):
        hits = hazard_entries(mission.trajectory[: step + 1], hazards)
        title.markdown(f"### {LABELS[mission.mode]}\n{'🔴' if hits else '🟢'} hazards hit: **{hits}**")
        image.image(render(world, mission, step, show_truth), width="stretch")


if play:
    for s in range(0, last + 1, 2):
        draw(s)
        time.sleep(0.04)
draw(step)

st.divider()
left, mid, rgt = st.columns([1.2, 1, 1])
with left:
    st.subheader("Results on this world")
    rows = []
    for mode, mission in missions.items():
        sc = score(world, mission.trajectory, mission.reached)
        rows.append({"rover": LABELS[mode], "reached": sc["reached"], "hazard entries": sc["hazard_entries"],
                     "success": sc["success"], "driven (m)": sc["path_m"], "stop": mission.stop_reason})
    st.dataframe(rows, hide_index=True, width="stretch")
with mid:
    st.subheader("Route decision")
    smart = missions["model"]
    card = decision_card(world, smart.belief, world.start, world.goal, smart.trajectory)
    st.info(card["summary"])
    st.caption(f"driven {card['chosen_m']} m vs terrain-blind shortest {card['shortest_m']} m "
               "(computed from the rover's own belief, not the true map)")
with rgt:
    st.subheader("Rover camera + model")
    cell = pair[1].frames[min(step, len(pair[1].frames) - 1)].cell
    st.image(camera(world, cell), caption=f"Curiosity NAVCAM tile {world.tile_at(cell)} with model overlay",
             width="stretch")
