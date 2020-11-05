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

DISCOUNT = 0.999
NUM_EPISODES = 300000
EPSILON_MIN = 0.1

Q = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))
state_visits = defaultdict(int)
np.random.seed(constants.SEED)

epsilon_decay = Decay(1, EPSILON_MIN, NUM_EPISODES)

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
            action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]

        next_state, reward, done = game.step(action)
        total_reward += reward

        Q[state.tobytes()][constants.BABY_MOVEMENTS.index(action)] += (
            1 / state_visits[state.tobytes()]
        ) * (
            reward
            + DISCOUNT * Q[next_state.tobytes()][constants.BABY_MOVEMENTS.index(action)]
            - Q[state.tobytes()][constants.BABY_MOVEMENTS.index(action)]
        )

        state = next_state.copy()
        state_visits[state.tobytes()] += 1
        steps += 1

    epsilon_decay.decay()

    episode_durations.append(steps)
    episode_rewards.append(total_reward)
    unique_states_seen.append(len(Q.keys()))

    if i % (NUM_EPISODES / 10) == 0:
        state, _, done = game.reset()
        while not done:
            action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
            state, reward, done = game.step(action, render=True)
            print(state)
            for pair in zip(constants.BABY_MOVEMENTS, Q[state.tobytes()]):
                print(f"{pair[0]}: {round(pair[1], 3)}", end=", ")
            print("\n")
        print(
            f"Average reward over last 50 episodes: {np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}"
        )

    if np.mean(episode_rewards[-constants.EPISODE_WINDOW :]) > 0:
        print(
            f"Game beaten in {i} episodes with average episode length over past ",
            f"{constants.EPISODE_WINDOW} episodes of ",
            f"{np.mean(episode_durations[-constants.EPISODE_WINDOW:])}",
        )
        break


with open("../policies/sarsa_policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

save_episode_duration_graph(
    "../images/epsiode_durations_sarsa.png",
    episode_durations,
    learner="Sarsa",
    mean_length=constants.EPISODE_WINDOW,
)

save_episode_reward_graph(
    "../images/episode_rewards_sarsa.png",
    episode_rewards,
    learner="Sarsa",
    mean_length=constants.EPISODE_WINDOW,
)

save_unique_states_graph(
    "../images/unique_states_sarsa.png", unique_states_seen, learner="Sarsa"
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

