import sys
from typing import Union

sys.path.append("..")

from utils import constants

import matplotlib.pyplot as plt
import matplotlib
import numpy as np


matplotlib.style.use("seaborn-dark")


def save_episode_duration_graph(
    filename: str,
    durations: list,
    learner: str,
    beaten: Union[int, bool],
    mean_length: int = 10,
) -> None:
    """Creates and saves a plot of episode durations average over the mean length
    puts a vertical line in where the game was beaten for the first time

    Parameters
    ----------
    filename : str
        name of the file to be saved with its location
    durations : list
        list of the duration of each episode over training
    learner : str
        name of the learner
    beaten : int or bool
        an integer of which episode beat the game for the first time, or a bool = False
        showing that the game was never beaten
    mean_length : int, optional
        the period over which to calculate running averages, by default 10
    """
    assert beaten > 0 or beaten == False, "beaten must be a positive integer or False"
    assert mean_length > 0, "mean length must be a positive integer"
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
    if beaten:
        plt.axvline(x=beaten)
        plt.text(x=beaten + 5, y=min(means) + 10, s="Game beaten", rotation=90)
    ax.grid()
    fig.savefig(filename)


def save_episode_reward_graph(
    filename: str,
    rewards: list,
    learner: str,
    proportion_decay_over: float,
    episodes: int,
    beaten: int,
    mean_length: int = 10,
) -> None:
    """Creates and saves a plot of episode durations average over the mean length
    puts a vertical line in where the game was beaten for the first time. Also puts
    a vertical line where epsilon was decayed to its minimal value

    Parameters
    ----------
    filename : str
        name of the file to be saved with its location
    rewards : list
        list of episode rewards
    learner : str
        name of the learner
    proportion_decay_over : float
        proportion as a decimal for which epsilon was decayed over
    episodes : int
        number of episodes that were used for training
    beaten : int
        integer for which the game was beaten first or False showing game was not beaten
    mean_length : int, optional
        length to calculate the running average over, by default 10
    """
    assert (
        0 < proportion_decay_over <= 1
    ), "proportion to decay over should be a decimal between 0 and 1"
    assert i > 0 or i == False, "beaten must be a positive integer or False"
    assert mean_length > 0, "mean length must be a positive integer"
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
    ax.grid()
    plt.axvline(x=proportion_decay_over * episodes)
    plt.text(
        x=proportion_decay_over * episodes + 5,
        y=min(means) + 10,
        s="Epsilon fully decayed",
        rotation=90,
    )
    if beaten:
        plt.axvline(x=beaten, color="r")
        plt.text(x=beaten + 5, y=min(means) + 10, s="Game beaten", rotation=90)
    fig.savefig(filename)


def save_unique_states_graph(filename: str, unique_states: list, learner: str) -> None:
    """Plots the cumulative number of unique states that have been seen at each
    stage of learning

    Parameters
    ----------
    filename : str
        Name of file to be saved and location
    unique_states : list
        List of unique states (cumulative) seen at each stage of learning
    learner : str
        Name of learner
    """
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
