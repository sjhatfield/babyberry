import sys

sys.path.append("..")

from environment.game import Game
from visualization.plots import (
    save_episode_duration_graph,
    save_episode_reward_graph,
    save_unique_states_graph,
)
from models.learning_utils import Decay, init_game_for_learning
import numpy as np
from collections import defaultdict
import pickle
from tqdm import tqdm
from utils import constants

DISCOUNT = 0.9
EPSILON_MIN = 0.01
PROPORTION_DECAY_EPSILON_OVER = 0.9
SMART_DAD = False
if SMART_DAD:
    folder = "smart_dad"
else:
    folder = "dumb_dad"

Q = defaultdict(lambda: [0] * (len(constants.BABY_MOVEMENTS) - 1) + [1])
state_visits = defaultdict(int)
np.random.seed(constants.SEED)

epsilon_decay = Decay(
    1,
    EPSILON_MIN,
    constants.EPISODES_TO_LEARN,
    proportion_to_decay_over=PROPORTION_DECAY_EPSILON_OVER,
)

game = init_game_for_learning(dumb_dad=~SMART_DAD)

episode_durations = []
episode_rewards = []
unique_states_seen = []

for i in tqdm(range(constants.EPISODES_TO_LEARN)):
    state, total_reward, done = game.reset()
    state_visits[state.tobytes()] += 1
    steps = 0
    while not done:
        action = epsilon_decay.select_action(state, Q)

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

    if i % (constants.EPISODES_TO_LEARN / 10) == 0:
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


with open(f"../policies/{folder}/sarsa/policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

save_episode_duration_graph(
    f"../images/{folder}/sarsa/episode_durations.png",
    episode_durations,
    learner="Sarsa",
    mean_length=constants.EPISODE_WINDOW,
)

save_episode_reward_graph(
    f"../images/{folder}/sarsa/episode_rewards.png",
    episode_rewards,
    learner="Sarsa",
    proportion_decay_over=PROPORTION_DECAY_EPSILON_OVER,
    mean_length=constants.EPISODE_WINDOW,
)

save_unique_states_graph(
    f"../images/{folder}/sarsa/unique_states.png", unique_states_seen, learner="Sarsa"
)


with open(f"../data/{folder}/Qlearner/rewards.pickle", "wb") as f:
    pickle.dump(episode_rewards, f)

with open(f"../data/{folder}/Qlearner/durations.pickle", "wb") as f:
    pickle.dump(episode_durations, f)
