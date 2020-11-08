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

DISCOUNT = 0.9
EPSILON_MIN = 0.01
Experience = namedtuple("Experience", field_names=["state", "action", "reward"])

np.random.seed(constants.SEED)

# Initialize the learning data structures
Q = defaultdict(lambda: [np.random.random()] * len(constants.BABY_MOVEMENTS))
state_visits = defaultdict(int)
returns = defaultdict(list)
epsilon_decay = Decay(1, EPSILON_MIN, constants.EPISODES_TO_LEARN, 0.75)

game = init_game_for_learning()

# Store episode statistics to measure success
episode_rewards = []
unique_states_seen = []
episode_durations = []

# Learning begins
for ep in tqdm(range(constants.EPISODES_TO_LEARN)):
    state, reward, done = game.reset()
    action = epsilon_decay.select_action(state, Q)
    episode_tuples = [Experience(state, action, reward)]
    episode_length = 1
    total_reward = reward

    # Generate an episode and store the (S, A, R) tuples
    while not done:
        state, reward, done = game.step(action)
        total_reward += reward
        episode_length += 1
        episode_tuples.append(Experience(state, action, reward))
        action = epsilon_decay.select_action(state, Q)

    # Update statistics
    episode_rewards.append(total_reward)
    episode_durations.append(episode_length)

    # Update Q-values based on the episode
    G = 0
    for i in range(len(episode_tuples) - 2, 0, -1):
        G = DISCOUNT * G + episode_tuples[i + 1].reward
        state_action_present = False
        for j in range(i):
            if (episode_tuples[i].state == episode_tuples[j].state).all():
                if episode_tuples[i].action == episode_tuples[j].action:
                    state_action_present = True
        if not state_action_present:
            returns[
                (episode_tuples[i].state.tobytes(), episode_tuples[i].action)
            ].append(G)
            Q[episode_tuples[i].state.tobytes()][
                constants.BABY_MOVEMENTS.index(episode_tuples[i].action)
            ] = np.mean(
                returns[(episode_tuples[i].state.tobytes(), episode_tuples[i].action)]
            )

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

# Save the Q-values which is the policy
with open("../policies/monte_carlo_ES/policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

# Save some graphs from the episode statistics
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

with open("../data/monte_carlo_ES/rewards.pickle", "wb") as f:
    pickle.dump(episode_rewards)

with open("../data/monte_carlo_ES/durations.pickle", "wb") as f:
    pickle.dump(episode_durations)
