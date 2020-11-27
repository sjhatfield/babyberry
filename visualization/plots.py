import sys

sys.path.append("..")

from utils import constants

import matplotlib.pyplot as plt
import matplotlib
import numpy as np


matplotlib.style.use("seaborn-dark")


def save_episode_duration_graph(
    filename: str, durations: list, learner: str, mean_length: int = 10
) -> None:
    fig, ax = plt.subplots(dpi=600)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode duration")
    ax.set_title(f"Episode duration mean over {mean_length} episodes for {learner}")
    means = [
        np.array(durations[x : x + mean_length]).mean()
        for x in range(len(durations) - mean_length)
    ]
    ax.plot(
        [i for i in range(mean_length, len(durations))],
        means,
        color="green",
        linewidth=2,
    )
    ax.annotate(
        f"{len(durations)} episodes",
        (len(durations) - 1, means[-1]),
        xytext=(len(durations) - 1 + 200, means[-1]),
    )
    ax.grid()
    fig.savefig(filename)


def save_episode_reward_graph(
    filename: str,
    rewards: list,
    learner: str,
    proportion_decay_over: float,
    mean_length: int = 10,
) -> None:
    fig, ax = plt.subplots(dpi=600)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode total reward")
    ax.set_title(f"Episode reward mean over {mean_length} episodes for {learner}")
    means = [
        np.array(rewards[x : x + mean_length]).mean()
        for x in range(len(rewards) - mean_length)
    ]
    ax.plot(
        [i for i in range(mean_length, len(rewards))],
        means,
        color="green",
        linewidth=2,
    )
    ax.annotate(
        f"{len(rewards)} episodes",
        (len(rewards) - 1, means[-1]),
        xytext=(len(rewards) - 1, means[-1] - 10),
    )
    ax.grid()
    plt.axvline(x=proportion_decay_over * constants.EPISODES_TO_LEARN)
    plt.text(
        x=proportion_decay_over * constants.EPISODES_TO_LEARN + 5,
        y=min(means) + 10,
        s="Epsilon fully decayed",
        rotation=90,
    )
    fig.savefig(filename)


def save_unique_states_graph(filename: str, unique_states: list, learner: str) -> None:
    fig, ax = plt.subplots(dpi=600)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Unique states")
    ax.set_title(f"Number of unique states seen for {learner}")
    ax.plot(
        [i for i in range(len(unique_states))],
        unique_states,
        color="green",
        linewidth=2,
    )
    ax.grid()
    fig.savefig(filename)
