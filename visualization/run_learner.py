import sys, pickle, argparse

parser = argparse.ArgumentParser(
    description="Save a video of learner playing the game and following the final policy."
)

parser.add_argument(
    "learner", choices=["Qlearner", "double_Qlearner", "nstep_sarsa", "sarsa", "random"]
)

parser.add_argument(
    "dumb",
    choices=[0, 1],
    help="1 means the dad is dumb and moves randomly. 0 means he is smart and moves towards the baby",
    type=int,
)

args = parser.parse_args()

sys.path.append("..")

from models.learning_utils import init_game_for_learning
from utils import constants

import cv2
import numpy as np

# np.random.seed(constants.SEED)

game = init_game_for_learning(dumb_dad=args.dumb)

if args.dumb:
    folder = "dumb_dad"
else:
    folder = "smart_dad"

print(folder)
if args.learner == "random":
    state, reward, done = game.reset()
    first = True
    while not done:
        action = np.random.choice(constants.BABY_MOVEMENTS)
        state, reward, done = game.step(action, True)
        if first:
            input("waiting for recorder")
            first = False
else:
    with open(f"../policies/{folder}/{args.learner}/policy.pickle", "rb") as f:
        Q = pickle.load(f)

    state, total_reward, done = game.reset()
    first = True
    while not done:
        print(state)
        try:
            for action, value in zip(constants.BABY_MOVEMENTS, Q[state.tobytes()]):
                print(f"{action=}, {value=}")
            action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
        except:
            print("took random action")
            action = np.random.choice(constants.BABY_MOVEMENTS)
        state, reward, done = game.step(action, True)
        total_reward += reward
        print(f"{reward=}")
        if first:
            input("waiting for recorder")
            first = False

print(f"{total_reward=}")

