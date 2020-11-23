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
from collections import defaultdict, namedtuple
import pickle
from tqdm import tqdm

# Learning hyperparameters
DISCOUNT = 0.9
EPSILON_MIN = 0.01
PROPORTION_DECAY_EPSILON_OVER = 1
Experience = namedtuple(
    "Experience", field_names=["state", "action", "greedy", "reward"]
)

np.random.seed(constants.SEED)

# Initialize the learning data structures
Q = defaultdict(lambda: [np.random.random()] * len(constants.BABY_MOVEMENTS))
C = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))
state_visits = defaultdict(int)
epsilon_decay = Decay(
    1,
    EPSILON_MIN,
    constants.EPISODES_TO_LEARN,
    proportion_to_decay_over=PROPORTION_DECAY_EPSILON_OVER,
)

game = init_game_for_learning()

# Store episode statistics to measure success
episode_rewards = []
unique_states_seen = []
episode_durations = []

# Learning begins
for ep in tqdm(range(constants.EPISODES_TO_LEARN)):
    state, reward, done = game.reset()
    total_reward = 0
    episode_tuples = []
    episode_length = 0

    # Generate an episode and store the (S, A, R) tuples
    while not done:
        action, greedy = epsilon_decay.select_action_MC(state, Q)
        new_state, reward, done = game.step(action)
        episode_tuples.append(Experience(state, action, greedy, reward))
        total_reward += reward
        episode_length += 1
        state = new_state

    # Update statistics
    episode_rewards.append(total_reward)
    episode_durations.append(episode_length)

    # Update Q-values based on the episode
    G = 0
    W = 1
    # Loop over episode in reverse oreder
    for i in range(len(episode_tuples) - 1, 0, -1):
        # Get the tuple information
        S_t = episode_tuples[i].state
        A_t = episode_tuples[i].action
        greedy_t = episode_tuples[i].greedy
        R_t = episode_tuples[i].reward
        A_t_index = constants.BABY_MOVEMENTS.index(A_t)

        G = DISCOUNT * G + R_t
        C[S_t.tobytes()][A_t_index] += W
        Q[S_t.tobytes()][A_t_index] += (W / C[S_t.tobytes()][A_t_index]) * (
            G - Q[S_t.tobytes()][A_t_index]
        )
        # If action was not greedy then end learning from episode
        if not greedy_t:
            break

        # If action was greedy then update weight W with probability of
        # greedy action being selected
        current_epsilon = epsilon_decay.get_current_value()
        prob_greedy_action = current_epsilon + (
            current_epsilon / len(constants.BABY_MOVEMENTS)
        )
        W = W / prob_greedy_action

    unique_states_seen.append(len(Q.keys()))
    epsilon_decay.decay()

    # Print some progress to CLI
    if ep % (constants.EPISODES_TO_LEARN / 10) == 0:
        print(
            f"Average reward over last {constants.EPISODE_WINDOW} episodes: {np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}"
        )

    # Check for game completion
    if np.mean(episode_rewards[-constants.EPISODE_WINDOW :]) > constants.WIN_AVERAGE:
        print(
            f"Game beaten in {i} episodes with average episode length over past ",
            f"{constants.EPISODE_WINDOW} episodes of ",
            f"{np.mean(episode_durations[-constants.EPISODE_WINDOW:])}",
        )
        break

# Save the policy
with open("../policies/off_policy_monte_carlo_control/policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

# Save some graphs related to learing
save_episode_duration_graph(
    "../images/off_policy_monte_carlo_control/episode_durations.png",
    episode_durations,
    learner="Off-policy Monte Carlo Control",
    mean_length=constants.EPISODE_WINDOW,
)

# Save the Q-values which is the policy
save_episode_reward_graph(
    "../images/off_policy_monte_carlo_control/episode_rewards.png",
    episode_rewards,
    learner="Off-policy Monte Carlo Control)",
    mean_length=constants.EPISODE_WINDOW,
)

# Save some graphs from the episode statistics
save_unique_states_graph(
    "../images/off_policy_monte_carlo_control/unique_states.png",
    unique_states_seen,
    learner="Off-policy Monte Carlo Control",
)

# Save the rewards and durations
with open("../data/off_policy_monte_carlo_control/rewards.pickle", "wb") as f:
    pickle.dump(episode_rewards, f)

with open("../data/off_policy_monte_carlo_control/durations.pickle", "wb") as f:
    pickle.dump(episode_durations, f)
