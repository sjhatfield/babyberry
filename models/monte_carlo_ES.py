import sys

sys.path.append("..")

from environment.game import Game
from visualization.plots import (
    save_episode_duration_graph,
    save_episode_reward_graph,
    save_unique_states_graph,
)
from models.decay import Decay
import numpy as np
from collections import defaultdict
import pickle
from tqdm import tqdm
from utils import constants
from collections import defaultdict
import time

DISCOUNT = 0.9

Q = defaultdict(lambda: [np.random.random()] * len(constants.BABY_MOVEMENTS))
state_visits = defaultdict(int)
np.random.seed(constants.SEED)

game = Game(
    board_size=9,
    baby_initial_position=[4, 4],
    move_reward=-1,
    eat_reward=5,
    illegal_move_reward=-100,
    complete_reward=100,
    num_berries=5,
    berry_movement_probabilities=[0.5] * 5,
    state_size=constants.STATE_SIZE,
    dad_initial_position=-1,
    dad_movement_probability=constants.DEFAULT_MOVEMENT_PROBABILITY,
)

episode_rewards = []
unique_states_seen = []
episode_durations = []
episode_rewards_mean = -1e4

returns = defaultdict(list)

start_time = time.time()

ep = 0
while episode_rewards_mean <= 0 and ep < 300000:
    state, total_reward, done = game.reset()
    action = np.random.choice(constants.BABY_MOVEMENTS)
    episode_tuples = [(state, action, total_reward)]
    episode_length = 1
    while not done:
        state, reward, done = game.step(action)
        total_reward += reward
        episode_length += 1
        episode_tuples.append((state, action, done))
        action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
    episode_rewards.append(total_reward)
    episode_durations.append(episode_length)
    G = 0
    for i in range(len(episode_tuples) - 2, 0, -1):
        G = DISCOUNT * G + episode_tuples[i + 1][2]
        state_action_present = False
        for j in range(i):
            if (episode_tuples[i][0] == episode_tuples[j][0]).all():
                if episode_tuples[i][1] == episode_tuples[j][1]:
                    state_action_present = True
        if not state_action_present:
            returns[(episode_tuples[i][0].tobytes(), episode_tuples[i][1])].append(G)
            Q[episode_tuples[i][0].tobytes()][
                constants.BABY_MOVEMENTS.index(episode_tuples[i][1])
            ] = np.mean(returns[(episode_tuples[i][0].tobytes(), episode_tuples[i][1])])
    if len(episode_rewards) >= constants.EPISODE_WINDOW:
        episode_rewards_mean = np.mean(episode_rewards[-constants.EPISODE_WINDOW :])
    ep += 1
    if ep % 200 == 0:
        print(f"Completed {ep} episodes of learning")
        print(
            f"Current average reward over last {constants.EPISODE_WINDOW} episodes is {episode_rewards_mean}"
        )
    unique_states_seen.append(len(Q.keys()))

print(f"Learning took {round(time.time() - start_time, 3)} seconds")

with open("../policies/monte_carlo_ES_policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

save_episode_duration_graph(
    "../images/epsiode_durations_monte_carlo_ES.png",
    episode_durations,
    learner="Monte Carlos (Exploring Starts)",
    mean_length=constants.EPISODE_WINDOW,
)

save_episode_reward_graph(
    "../images/episode_rewards_monte_carlo_ES.png",
    episode_rewards,
    learner="Monte Carlos (Exploring Starts)",
    mean_length=constants.EPISODE_WINDOW,
)

save_unique_states_graph(
    "../images/unique_states_monte_carlo_ES.png",
    unique_states_seen,
    learner="Monte Carlos (Exploring Starts)",
)

# Final demonstration of learner
state, _, done = game.reset()
while not done:
    action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
    state, reward, done = game.step(action, render=True)
    print(state)
    for pair in zip(constants.BABY_MOVEMENTS, Q[state.tobytes()]):
        print(f"{pair[0]}: {round(pair[1], 3)}", end=", ")
    print("\n")

print(f"Number of unique states seen: {len(Q.keys())}")
