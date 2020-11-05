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
from operator import add
from tqdm import tqdm
from utils import constants

DISCOUNT = 0.9
NUM_EPISODES = 10000
EPSILON_MIN = 0.1

Q1 = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))
Q2 = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))
Q = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))
state_visits = defaultdict(int)
np.random.seed(constants.SEED)

epsilon_decay = Decay(1, EPSILON_MIN, NUM_EPISODES)


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

episode_durations = []
episode_rewards = []
unique_states_seen = []

for i in tqdm(range(NUM_EPISODES)):
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

    if i % (NUM_EPISODES / 10) == 0:
        print(
            f"Average reward over last 50 episodes: {np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}"
        )

    if np.mean(episode_rewards[-constants.EPISODE_WINDOW :]) > constants.WIN_AVERAGE:
        print(
            f"Game beaten in {i} episodes with average episode length over past ",
            f"{constants.EPISODE_WINDOW} episodes of ",
            f"{np.mean(episode_durations[-constants.EPISODE_WINDOW:])}",
        )
        break


with open("../policies/double_Qlearner/policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

save_episode_duration_graph(
    "../images/double_Qlearner/epsiode_durations.png",
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

""" # Final demonstration of learner using sum of two Q values
state, _, done = game.reset()
while not done:
    action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
    state, reward, done = game.step(action, render=True)
    print(state)
    for pair in zip(constants.BABY_MOVEMENTS, Q[state.tobytes()]):
        print(f"{pair[0]}: {round(pair[1], 3)}", end=", ")
    print("\n") """

print(f"Number of unique states seen: {len(Q.keys())}")
