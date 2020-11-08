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
from tqdm import tqdm

DISCOUNT = 0.9
NUM_EPISODES = 30000
EPSILON_MIN = 0.01
N = 10

np.random.seed(constants.SEED)

# Initialize the learning data structures
Q = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))
state_visits = defaultdict(int)
epsilon_decay = Decay(1, EPSILON_MIN, NUM_EPISODES, proportion_to_decay_over=0.75)
states = [0] * N
actions = [0] * N
rewards = [0] * N

game = init_game_for_learning()

# Store episode statistics to measure success
episode_durations = []
episode_rewards = []
unique_states_seen = []

# Begin learning
for i in tqdm(range(NUM_EPISODES)):
    # Return to beginning of game
    state, total_reward, done = game.reset()
    states[0] = state
    state_visits[state.tobytes()] += 1
    steps = 0

    # Select and store first action
    action = epsilon_decay.select_action(state, Q)
    actions[0] = action
    T = np.inf
    t = 0
    tau = -1

    # Perform updates the Q
    while tau != (T - 1):
        # Take next step in episode and store
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
                action = epsilon_decay.select_action(state, Q)
                actions[(t + 1) % N] = action
        tau = t - N + 1
        if tau >= 0:
            G = sum(
                [(DISCOUNT ** (j - tau - 1)) * rewards[j % N]]
                for j in range(tau + 1, min(tau + N, T) + 1)
            )
            # If steps do not take us beyond the end of the episode
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
        t += 1

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
            f"{np.mean(episode_durations[-constants.EPISODE_WINDOW:])}",
        )
        break

# Save the Q-values which is the policy
with open("../policies/nstep_sarsa/policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

# Save some graphs from the episode statistics
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

