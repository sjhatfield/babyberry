import sys

sys.path.append("..")

from environment.game import Game
from visualization.plots import (
    save_episode_duration_graph,
    save_episode_reward_graph,
    save_unique_states_graph,
)
from models.learning_utils import Decay, init_game_for_learning, sample_action
from utils import constants

import numpy as np
from collections import defaultdict
import pickle
from tqdm import tqdm

# Learning hyperparameters
DISCOUNT = 0.9
NUM_EPISODES = 10000
EPSILON_MIN = 0.01
N = 10
PROPORTION_DECAY_EPSILON_OVER = 1

np.random.seed(constants.SEED)

# Initialize the learning data structures
Q = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))
E = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))  # Eligibility traces
state_visits = defaultdict(int)

epsilon_decay = Decay(
    1, EPSILON_MIN, NUM_EPISODES, proportion_to_decay_over=PROPORTION_DECAY_EPSILON_OVER
)

game = init_game_for_learning()

episode_durations = []
episode_rewards = []
unique_states_seen = []
sigma = 0  # will alternate between 0 and 1 each time step

# Begin learning
for i in tqdm(range(NUM_EPISODES)):
    # Return to beginning of the game
    state, total_reward, done = game.reset()
    steps = 0
    state_visits[state.tobytes()] += 1

    # Select and store the first action
    policy = epsilon_decay.get_policy(Q, state)
    action = sample_action(policy)
    actions = [action]

    while not done:
        steps += 1
        sigma = (sigma + 1) % 2
        next_state, reward, done = game.step(action)
        state_visits[next_state.tobytes()] += 1
        total_reward += reward
        policy = epsilon_decay.get_policy(Q, next_state)
        next_action = sample_action(policy)

        policy = epsilon_decay.get_policy(Q, next_state)

        sarsa_target = Q[next_state.tobytes()][
            constants.BABY_MOVEMENTS.index(next_action)
        ]
        exp_sarsa_target = np.dot(policy, Q[next_state.tobytes()])
        td_target = reward + DISCOUNT * (
            sigma * sarsa_target + (1 - sigma) * exp_sarsa_target
        )
        td_error = (
            td_target - Q[state.tobytes()][constants.BABY_MOVEMENTS.index(action)]
        )

        E[state.tobytes()][constants.BABY_MOVEMENTS.index(action)] += 1

        for state in Q:
            for i in range(len(constants.BABY_MOVEMENTS)):
                Q[state][i] += (1 / state_visits[state]) * E[state][i] * td_error
                E[state][i] = DISCOUNT * (
                    sigma
                    + policy[constants.BABY_MOVEMENTS.index(next_action)] * (1 - sigma)
                )

        if steps > 1000:
            break

        state = next_state
        action = next_action

    epsilon_decay.decay()

    episode_durations.append(steps)
    episode_rewards.append(total_reward)
    unique_states_seen.append(len(Q.keys()))

    # Print some progress to CLI
    if i % (NUM_EPISODES / 10) == 0:
        print(
            f"Average reward over last {constants.EPISODE_WINDOW} episodes: {np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}"
        )

    # Check for game completion
    if np.mean(episode_rewards[-constants.EPISODE_WINDOW :]) > 0:
        print(
            f"Game beaten in {i} episodes with average episode length over past ",
            f"{constants.EPISODE_WINDOW} episodes of ",
            f"{np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}",
        )
        break

# Save the policy
with open("../policies/Q_sigma/policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

# Print some graphs showing learning progress
save_episode_duration_graph(
    "../images/Q_sigma/episode_durations.png",
    episode_durations,
    learner="Q Sigma",
    mean_length=constants.EPISODE_WINDOW,
)

save_episode_reward_graph(
    "../images/Q_sigma/episode_rewards.png",
    episode_rewards,
    learner="Q Sigma",
    mean_length=constants.EPISODE_WINDOW,
)

save_unique_states_graph(
    "../images/Q_sigma/unique_states.png", unique_states_seen, learner="Sarsa"
)

# Save the rewards and durations
with open("../data/Q_sigma/rewards.pickle", "wb") as f:
    pickle.dump(episode_rewards, f)

with open("../data/Q_sigma/durations.pickle", "wb") as f:
    pickle.dump(episode_durations, f)

