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

DISCOUNT = 0.9
NUM_EPISODES = 30000
EPSILON_MIN = 0.01
N = 10

Q = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))
state_visits = defaultdict(int)
np.random.seed(constants.SEED)

epsilon_decay = Decay(1, EPSILON_MIN, NUM_EPISODES, proportion_to_decay_over=0.75)

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
states = [0] * N
actions = [0] * N
rewards = [0] * N

for i in tqdm(range(NUM_EPISODES)):
    state, total_reward, done = game.reset()
    states[0] = state
    state_visits[state.tobytes()] += 1
    steps = 0
    if np.random.random() < epsilon_decay.get_current_value():
        action = np.random.choice(constants.BABY_MOVEMENTS)
    else:
        action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
    actions[0] = action
    T = np.inf
    for t in range(int(1e6)):
        if t < T:
            state, reward, done = game.step(action)
            state_visits[state.tobytes()] += 1
            steps += 1
            total_reward += reward
            states[t % N] = state
            rewards[t % N] = reward
            if done:
                T = t + 1
            else:
                if np.random.random() < epsilon_decay.get_current_value():
                    action = np.random.choice(constants.BABY_MOVEMENTS)
                else:
                    action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
                actions[(t + 1) % N] = action
        tau = t - N + 1
        if tau >= 0:
            G = 0
            for j in range(tau + 1, min(tau + N, T) + 1):
                G += (DISCOUNT ** (j - tau - 1)) * rewards[j % N]
            if tau + N < T:
                G += (DISCOUNT ** N) * Q[states[(tau + N) % N].tobytes()][
                    constants.BABY_MOVEMENTS.index(actions[(tau + N) % N])
                ]
            Q[states[tau % N].tobytes()][
                constants.BABY_MOVEMENTS.index(actions[tau % N])
            ] += (1 / state_visits[states[tau % N].tobytes()]) * (
                G
                - Q[states[tau % N].tobytes()][
                    constants.BABY_MOVEMENTS.index(actions[tau % N])
                ]
            )
        if tau == (T - 1):
            break

    epsilon_decay.decay()

    episode_durations.append(steps)
    episode_rewards.append(total_reward)
    unique_states_seen.append(len(Q.keys()))

    if i % (NUM_EPISODES / 10) == 0:
        print(
            f"Average reward over last {constants.EPISODE_WINDOW} episodes: {np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}"
        )

    if np.mean(episode_rewards[-constants.EPISODE_WINDOW :]) > 0:
        print(
            f"Game beaten in {i} episodes with average episode length over past ",
            f"{constants.EPISODE_WINDOW} episodes of ",
            f"{np.mean(episode_durations[-constants.EPISODE_WINDOW:])}",
        )
        break


with open("../policies/nstep_sarsa/policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

save_episode_duration_graph(
    "../images/nstep_sarsa/episode_durations.png",
    episode_durations,
    learner="N-step Sarsa",
    mean_length=constants.EPISODE_WINDOW,
)

save_episode_reward_graph(
    "../images/nstep_sarsa/episode_rewards.png",
    episode_rewards,
    learner="N-step Sarsa",
    mean_length=constants.EPISODE_WINDOW,
)

save_unique_states_graph(
    "../images/nstep_sarsa/unique_states.png", unique_states_seen, learner="Sarsa"
)

print(f"Number of unique states seen: {len(Q.keys())}")

