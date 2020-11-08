import sys

sys.path.append("..")

from environment.game import Game
from visualization.plots import (
    save_episode_duration_graph,
    save_episode_reward_graph,
    save_unique_states_graph,
)
from models.learning_utils import Decay, init_game_for_learning
from utils import constants

import numpy as np
from collections import defaultdict
import pickle
from operator import add
from tqdm import tqdm


DISCOUNT = 0.9
EPSILON_MIN = 0.01

np.random.seed(constants.SEED)

Q1 = defaultdict(lambda: [0] * (len(constants.BABY_MOVEMENTS) - 1) + [1])
Q2 = defaultdict(lambda: [0] * (len(constants.BABY_MOVEMENTS) - 1) + [1])
Q = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))
state_visits = defaultdict(int)
epsilon_decay = Decay(
    1, EPSILON_MIN, constants.EPISODES_TO_LEARN, proportion_to_decay_over=0.2
)

# Need a way of generating actions and values from the two
# Q-value data structures
def get_action(Q1, Q2):
    Q_value = get_Q_value(Q1, Q2)
    return constants.BABY_MOVEMENTS[np.argmax(Q_value)]


def get_Q_value(Q1, Q2):
    Q_value = [0] * 5
    if state.tobytes() in Q1.keys():
        Q_value = list(map(add, Q_value, Q1[state.tobytes()]))
    if state.tobytes() in Q2.keys():
        Q_value = list(map(add, Q_value, Q2[state.tobytes()]))
    return Q_value


game = init_game_for_learning()

# Store episode statistics to measure success
episode_durations = []
episode_rewards = []
unique_states_seen = []

# Learning begins
for i in tqdm(range(constants.EPISODES_TO_LEARN)):
    state, total_reward, done = game.reset()
    state_visits[state.tobytes()] += 1
    steps = 0
    while not done:
        if np.random.random() < epsilon_decay.get_current_value():
            action = np.random.choice(constants.BABY_MOVEMENTS)
        else:
            action = get_action(Q1, Q2)

        next_state, reward, done = game.step(action)
        total_reward += reward

        if np.random.random() < 0.5:
            Q1[state.tobytes()][constants.BABY_MOVEMENTS.index(action)] += (
                1 / state_visits[state.tobytes()]
            ) * (
                reward
                + DISCOUNT
                * Q2[next_state.tobytes()][np.argmax(Q1[next_state.tobytes()])]
                - Q1[state.tobytes()][constants.BABY_MOVEMENTS.index(action)]
            )
        else:
            Q2[state.tobytes()][constants.BABY_MOVEMENTS.index(action)] += (
                1 / state_visits[state.tobytes()]
            ) * (
                reward
                + DISCOUNT
                * Q1[next_state.tobytes()][np.argmax(Q2[next_state.tobytes()])]
                - Q2[state.tobytes()][constants.BABY_MOVEMENTS.index(action)]
            )

        state = next_state.copy()
        state_visits[state.tobytes()] += 1
        steps += 1

        # Update the overall Q values
        Q[state.tobytes()][constants.BABY_MOVEMENTS.index(action)] = (
            Q1[state.tobytes()][constants.BABY_MOVEMENTS.index(action)]
            + Q2[state.tobytes()][constants.BABY_MOVEMENTS.index(action)]
        )

    epsilon_decay.decay()

    episode_durations.append(steps)
    episode_rewards.append(total_reward)
    unique_states_seen.append(max(len(Q1.keys()), len(Q2.keys())))

    if i % (constants.EPISODES_TO_LEARN / 10) == 0:
        print(
            f"Average reward over last {constants.EPISODE_WINDOW} episodes: {np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}"
        )

    if np.mean(episode_rewards[-constants.EPISODE_WINDOW :]) > constants.WIN_AVERAGE:
        print(
            f"Game beaten in {i} episodes with average episode length over past ",
            f"{constants.EPISODE_WINDOW} episodes of ",
            f"{np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}",
        )
        break


with open("../policies/double_Qlearner/policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

save_episode_duration_graph(
    "../images/double_Qlearner/episode_durations.png",
    episode_durations,
    learner="Double QLearner",
    mean_length=constants.EPISODE_WINDOW,
)

save_episode_reward_graph(
    "../images/double_Qlearner/episode_rewards.png",
    episode_rewards,
    learner="Double QLearner",
    mean_length=constants.EPISODE_WINDOW,
)

save_unique_states_graph(
    "../images/double_Qlearner/unique_states.png",
    unique_states_seen,
    learner="Double QLearner",
)

with open("../data/double_Qlearner/rewards.pickle", "wb") as f:
    pickle.dump(episode_rewards, f)

with open("../data/double_Qlearner/durations.pickle", "wb") as f:
    pickle.dump(episode_durations, f)
