import sys

sys.path.append("..")

from utils import constants
from models.learning_utils import init_game_for_learning

import pickle
import numpy as np

np.random.seed(constants.SEED)

try:
    learner = sys.argv[1]
    with open(f"../policies/{learner}/policy.pickle", "rb") as f:
        Q = pickle.load(f)
except Exception:
    print(f"Failed to load {learner=}")

try:
    num_iterations = int(sys.argv[2])
except Exception:
    print(f"Number of iterations is not an integer")

game = init_game_for_learning()

for i in range(num_iterations):
    state, reward, done = game.reset()
    steps = 0
    random_actions_needed = 0
    total_reward = reward
    while not done:
        try:
            action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
        except KeyError:
            action = np.random.choice(constants.BABY_MOVEMENTS)
            random_actions_needed += 1
        state, reward, done = game.step(action, True)
        total_reward += reward
        print(f"{reward=}")
        steps += 1
    print(f"{total_reward=}, {steps=}, {random_actions_needed=}")

