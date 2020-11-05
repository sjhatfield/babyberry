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
NUM_EPISODES = 10000
EPSILON_MIN = 0.01

Q = defaultdict(lambda: [np.random.random()] * len(constants.BABY_MOVEMENTS))
state_visits = defaultdict(int)
np.random.seed(constants.SEED)

epsilon_decay = Decay(1, EPSILON_MIN, NUM_EPISODES, 0.5)

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

returns = defaultdict(list)

start_time = time.time()

for ep in tqdm(range(NUM_EPISODES)):
    state, total_reward, done = game.reset()
    if np.random.random() < epsilon_decay.get_current_value():
        action = np.random.choice(constants.BABY_MOVEMENTS)
    else:
        action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
    episode_tuples = [(state, action, total_reward)]
    episode_length = 1
    while not done:
        state, reward, done = game.step(action)
        total_reward += reward
        episode_length += 1
        episode_tuples.append((state, action, done))
        if np.random.random() < epsilon_decay.get_current_value():
            action = np.random.choice(constants.BABY_MOVEMENTS)
        else:
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

    if ep % (NUM_EPISODES / 10) == 0:
        print(
            f"Average reward over last {constants.EPISODE_WINDOW} episodes: {np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}"
        )

    if np.mean(episode_rewards[-constants.EPISODE_WINDOW :]) > constants.WIN_AVERAGE:
        print(
            f"Game beaten in {i} episodes with average episode length over past ",
            f"{constants.EPISODE_WINDOW} episodes of ",
            f"{np.mean(episode_durations[-constants.EPISODE_WINDOW:])}",
        )
        break

    unique_states_seen.append(len(Q.keys()))
    epsilon_decay.decay()

print(f"Learning took {round(time.time() - start_time, 3)} seconds")

with open("../policies/monte_carlo_ES/policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

save_episode_duration_graph(
    "../images/monte_carlo_ES/episode_durations.png",
    episode_durations,
    learner="Monte Carlos (Exploring Starts)",
    mean_length=constants.EPISODE_WINDOW,
)

save_episode_reward_graph(
    "../images/monte_carlo_ES/episode_rewards.png",
    episode_rewards,
    learner="Monte Carlos (Exploring Starts)",
    mean_length=constants.EPISODE_WINDOW,
)

save_unique_states_graph(
    "../images/monte_carlo_ES/unique_states.png",
    unique_states_seen,
    learner="Monte Carlo (Exploring Starts)",
)

print(f"Number of unique states seen: {len(Q.keys())}")
