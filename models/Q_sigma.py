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

# Learning hyperparameters
DISCOUNT = 0.9
NUM_EPISODES = 30000
EPSILON_MIN = 0.01
N = 10
PROPORTION_DECAY_EPSILON_OVER = 1

np.random.seed(constants.SEED)

# Initialize the learning data structures
Q = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))
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
    states = [state]
    rewards = [0]
    state_visits[state.tobytes()] += 1
    steps = 0

    # Select and store the first action
    action = epsilon_decay.select_action(state, Q)
    actions = [action]

    sigmas = [0]
    rhos = [0]

    # T will be reassigned below, high to begin with
    T = np.inf

    #
    for t in range(int(1e6)):
        if t < T:
            # take next step in the episode
            state, reward, done = game.step(action)
            state_visits[state.tobytes()] += 1
            steps += 1
            total_reward += reward
            states.append(state)
            rewards.append(reward)

            # If episode over set T so that no more updates will happen
            if done:
                T = t + 1
            else:
                # Select random action
                action = np.random.choice(constants.BABY_MOVEMENTS)
                actions.append(action)
                sigma = (sigma + 1) % 2
                sigmas.append(sigma)
                current_epsilon = epsilon_decay.get_current_value()
                if action == constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]:
                    pi = (1 - current_epsilon) + current_epsilon / len(
                        constants.BABY_MOVEMENTS
                    )
                else:
                    pi = current_epsilon - current_epsilon / len(
                        constants.BABY_MOVEMENTS
                    )
                b = 1 / len(constants.BABY_MOVEMENTS)
                rhos.append(pi / b)

        # Set tau appropriately in the past
        tau = t - N + 1
        # If an update is possible
        if tau >= 0:
            if t + 1 < T:
                G = Q[state.tobytes()][constants.BABY_MOVEMENTS.index(action)]
            for k in range(min(t + 1, T), tau, -1):
                # if terminal
                if k == T:
                    G = rewards[T]
                # Otherwise find expected approximate value
                else:
                    V_hat = 0
                    for a in constants.BABY_MOVEMENTS:
                        if (
                            a
                            == constants.BABY_MOVEMENTS[
                                np.argmax(Q[states[k].tobytes()])
                            ]
                        ):
                            pi = (1 - current_epsilon) + current_epsilon / len(
                                constants.BABY_MOVEMENTS
                            )
                        else:
                            pi = 1 - (
                                (1 - current_epsilon)
                                + current_epsilon / len(constants.BABY_MOVEMENTS)
                            )
                        V_hat += (
                            pi
                            * Q[states[k].tobytes()][constants.BABY_MOVEMENTS.index(a)]
                        )
                    if (
                        actions[k]
                        == constants.BABY_MOVEMENTS[np.argmax(Q[states[k].tobytes()])]
                    ):
                        pi = (1 - current_epsilon) + current_epsilon / len(
                            constants.BABY_MOVEMENTS
                        )
                    else:
                        pi = 1 - (
                            (1 - current_epsilon)
                            + current_epsilon / len(constants.BABY_MOVEMENTS)
                        )
                    G = (
                        rewards[k]
                        + DISCOUNT
                        * (sigmas[k] * rhos[k] + (1 - sigmas[k]) * pi)
                        * (
                            G
                            - Q[states[k].tobytes()][
                                constants.BABY_MOVEMENTS.index(actions[k])
                            ]
                        )
                        + DISCOUNT * V_hat
                    )
            # Finally make the update to the Q-value
            Q[states[tau].tobytes()][constants.BABY_MOVEMENTS.index(actions[tau])] += (
                1 / state_visits[states[tau].tobytes()]
            ) * (
                G
                - Q[states[tau].tobytes()][constants.BABY_MOVEMENTS.index(actions[tau])]
            )
        if tau == (T - 1):
            break

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

