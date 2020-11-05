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
from collections import defaultdict
import time

DISCOUNT = 0.9
NUM_EPISODES = 30000

Q = defaultdict(lambda: [np.random.random()] * len(constants.BABY_MOVEMENTS))
C = defaultdict([0] * len(constants.BABY_MOVEMENTS))
state_visits = defaultdict(int)
np.random.seed(constants.SEED)

epsilon_decay = Decay(1, 0.01, NUM_EPISODES, 0.5)

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

episode_rewards = []
unique_states_seen = []
episode_durations = []

start_time = time.time()

for ep in tqdm(range(NUM_EPISODES)):
    state, total_reward, done = game.reset()
    if np.random.random() < epsilon_decay.get_current_value():
        action = np.random.choice(constants.BABY_MOVEMENTS)
    else:
        action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
    episode_tuples = [(state, action, total_reward)]
    episode_length = 1
    while not done:
        state, reward, done = game.step(action)
        total_reward += reward
        episode_length += 1
        episode_tuples.append((state, action, done))
        if np.random.random() < epsilon_decay.get_current_value():
            action = np.random.choice(constants.BABY_MOVEMENTS)
        else:
            action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
    episode_rewards.append(total_reward)
    episode_durations.append(episode_length)
    G = 0
    W = 1
    for i in range(len(episode_tuples) - 2, 0, -1):
        S_t = episode_tuples[i][0]
        A_t = episode_tuples[i][1]
        R_t = episode_tuples[i][2]
        G = DISCOUNT * G + R_t
        C[S_t.tobytes()][constants.BABY_MOVEMENTS.index(A_t)] += W
        Q[S_t.tobytes()][constants.BABY_MOVEMENTS.index(A_t)] += (
            W / C[S_t.tobytes()][constants.BABY_MOVEMENTS.index(A_t)]
        ) * (G - Q[S_t.tobytes()][constants.BABY_MOVEMENTS.index(A_t)])
        if A_t != constants.BABY_MOVEMENTS[np.argmax(Q[S_t.tobytes()])]:
            break
        current_epsilon = epsilon_decay.get_current_value()
        W = W / (current_epsilon - (current_epsilon / 5))

    if len(episode_rewards) >= constants.EPISODE_WINDOW:
        episode_rewards_mean = np.mean(episode_rewards[-constants.EPISODE_WINDOW :])

    if ep % (NUM_EPISODES / 10) == 0:
        print(
            f"Average reward over last {constants.EPISODE_WINDOW} episodes: {np.mean(episode_rewards[-constants.EPISODE_WINDOW:])}"
        )

    if np.mean(episode_rewards[-constants.EPISODE_WINDOW :]) > constants.WIN_AVERAGE:
        print(
            f"Game beaten in {i} episodes with average episode length over past ",
            f"{constants.EPISODE_WINDOW} episodes of ",
            f"{np.mean(episode_durations[-constants.EPISODE_WINDOW:])}",
        )
        break

    unique_states_seen.append(len(Q.keys()))
    epsilon_decay.decay()

print(f"Learning took {round(time.time() - start_time, 3)} seconds")

with open("../policies/off_policy_monte_carlo_control/policy.pickle", "wb") as f:
    pickle.dump(dict(Q), f)

save_episode_duration_graph(
    "../images/off_policy_monte_carlo_control/episode_durations.png",
    episode_durations,
    learner="Off-policy Monde Carlo Control",
    mean_length=constants.EPISODE_WINDOW,
)

save_episode_reward_graph(
    "../images/off_policy_monte_carlo_control/episode_rewards.png",
    episode_rewards,
    learner="Off-policy Monde Carlo Control)",
    mean_length=constants.EPISODE_WINDOW,
)

save_unique_states_graph(
    "../images/off_policy_monte_carlo_control/unique_states.png",
    unique_states_seen,
    learner="Off-policy Monde Carlo Control",
)

print(f"Number of unique states seen: {len(Q.keys())}")
