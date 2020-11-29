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
N = 3  # Needs to be relatively low due to small window on state
SMART_DAD = False
if SMART_DAD:
    folder = "smart_dad"
else:
    folder = "dumb_dad"

np.random.seed(constants.SEED)

# Initialize the learning data structures
Q = defaultdict(lambda: [0] * len(constants.BABY_MOVEMENTS))
state_visits = defaultdict(int)
epsilon_decay = Decay(
    1,
    constants.EPSILON_MIN,
    constants.EPISODES_TO_LEARN[folder],
    proportion_to_decay_over=constants.PROPORTION_DECAY_EPSILON_OVER,
)

game = init_game_for_learning(dumb_dad=not SMART_DAD)

# Store episode statistics to measure success
episode_durations = []
episode_rewards = []
unique_states_seen = []
beaten = False

# Begin learning
for i in tqdm(range(constants.EPISODES_TO_LEARN[folder])):
    # Return to beginning of game
    state, total_reward, done = game.reset()
    states = [state]
    rewards = [0]
    state_visits[state.tobytes()] += 1
    steps = 0

    # Select and store first action
    action = epsilon_decay.select_action(state, Q)
    actions = [action]

    # T will be reassigned below, need it high to begin with
    T = np.inf
    t = 0
    tau = -1

    # Perform updates to Q
    while tau != (T - 1):

        # Take next step in episode and store
        if t < T:
            state, reward, done = game.step(action)
            state_visits[state.tobytes()] += 1
            steps += 1
            total_reward += reward
            states.append(state)
            rewards.append(reward)
            # If done set T so that no more steps are taken above
            if done:
                T = t + 1
            else:
                action = epsilon_decay.select_action(state, Q)
                actions.append(action)
        # tau is the step getting updated
        tau = t - N + 1
        # If an update is possible
        if tau >= 0:
            # Sum discounted past reward
            G = sum(
                [
                    (constants.DISCOUNT ** (j - tau - 1)) * rewards[j]
                    for j in range(tau + 1, min(tau + N, T) + 1)
                ]
            )
            # If steps do not take us beyond the end of the episode
            if tau + N < T:
                s = states[tau + N]
                a = actions[tau + N]
                G += (constants.DISCOUNT ** N) * Q[s.tobytes()][
                    constants.BABY_MOVEMENTS.index(a)
                ]
            s = states[tau]
            a = actions[tau]
            Q[s.tobytes()][constants.BABY_MOVEMENTS.index(a)] += (
                1 / state_visits[s.tobytes()]
            ) * (G - Q[s.tobytes()][constants.BABY_MOVEMENTS.index(a)])
        t += 1

    epsilon_decay.decay()

    episode_durations.append(steps)
    episode_rewards.append(total_reward)
    unique_states_seen.append(len(Q.keys()))

    # Print some progress to CLI
    if i % (constants.EPISODES_TO_LEARN[folder] / 10) == 0:
        print(
            f"Average reward over last {constants.EPISODE_WINDOW} episodes: {np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}"
        )

    # Check for game completion
    if not beaten:
        if (
            np.mean(episode_rewards[-constants.EPISODE_WINDOW :])
            > constants.WIN_AVERAGE[folder]
        ):
            print(
                f"Game beaten in {i} episodes with average episode length over past ",
                f"{constants.EPISODE_WINDOW} episodes of ",
                f"{np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}",
            )
            beaten = i

# Save the Q-values which is the policy
with open(f"../policies/{folder}/nstep_sarsa/policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

# Save some graphs from the episode statistics
save_episode_duration_graph(
    f"../images/{folder}/nstep_sarsa/episode_durations.png",
    episode_durations,
    learner="N-step Sarsa",
    mean_length=constants.EPISODE_WINDOW,
    beaten=beaten,
)

save_episode_reward_graph(
    f"../images/{folder}/nstep_sarsa/episode_rewards.png",
    episode_rewards,
    learner="N-step Sarsa",
    episodes=constants.EPISODES_TO_LEARN[folder],
    proportion_decay_over=constants.PROPORTION_DECAY_EPSILON_OVER,
    mean_length=constants.EPISODE_WINDOW,
    beaten=beaten,
)

save_unique_states_graph(
    f"../images/{folder}/nstep_sarsa/unique_states.png",
    unique_states_seen,
    learner="Sarsa",
)

# Save the rewards and durations
with open(f"../data/{folder}/nstep_sarsa/rewards.pickle", "wb") as f:
    pickle.dump(episode_rewards, f)

with open(f"../data/{folder}/nstep_sarsa/durations.pickle", "wb") as f:
    pickle.dump(episode_durations, f)
