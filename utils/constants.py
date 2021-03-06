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
WIN_AVERAGE = {"dumb_dad": 0, "smart_dad": -30}
EPISODES_TO_LEARN = {"dumb_dad": 30000, "smart_dad": 100000}
DISCOUNT = 0.9
EPSILON_MIN = 0.01
PROPORTION_DECAY_EPSILON_OVER = 0.9
