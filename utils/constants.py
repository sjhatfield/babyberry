# Environment
BABY_MOVEMENTS = ["N", "E", "S", "W", "R"]
BERRY_MOVEMENTS = ["N", "E", "S", "W"]
CHARACTER_KEY = {"baby": -1, "tile": 0, "edge": -2, "dad": -8}
DEFAULT_MOVEMENT_PROBABILITY = 0.5
COLORS = {
    "baby": (102, 230, 255),
    "tile": (255, 255, 255),
    "berry": (230, 0, 115),
    "dad": (0, 102, 51),
}
WAIT_MS = 400

# Constants across learners
SEED = 3142
STATE_SIZE = 5
EPISODE_WINDOW = 200
WIN_AVERAGE = 0
EPISODES_TO_LEARN = 30000
