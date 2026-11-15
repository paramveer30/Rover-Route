import numpy as np

from roverroute.card import decision_card
from roverroute.data import SAND
from roverroute.perception import Belief
from roverroute.world import SIZE, World


def flat_world():
    return World(0, np.zeros((SIZE, SIZE), np.uint8), np.zeros((SIZE, SIZE, 4), np.float32),
                 np.zeros((SIZE, SIZE)), [["t"] * 3] * 3, (30, 0), (30, 59))


def test_card_reports_avoided_sand():
    world, belief = flat_world(), Belief.empty()
    belief.seen[:] = belief.known[:] = True
    belief.probs[:] = [1, 0, 0, 0]
    belief.probs[25:36, 20:40] = [0.1, 0, 0.9, 0]
    belief.unc[:] = 0
    detour = [(30, c) for c in range(0, 19)] + [(r, 19) for r in range(30, 23, -1)]
    detour += [(24, c) for c in range(20, 41)] + [(r, 41) for r in range(24, 31)] + [(30, c) for c in range(42, 60)]
    card = decision_card(world, belief, world.start, world.goal, detour)
    assert card["avoided_class"] == "sand" and card["avoided_cells"] > 0 and card["extra_m"] > 0
    assert card["summary"].startswith("Took a") and "sand" in card["summary"]
    assert belief.probs[30, 30].argmax() == SAND


def test_card_when_route_is_shortest():
    world, belief = flat_world(), Belief.empty()
    straight = [(30, c) for c in range(60)]
    card = decision_card(world, belief, world.start, world.goal, straight)
    assert card["extra_m"] == 0 and card["summary"] == "Took the shortest route; nothing needed avoiding."
