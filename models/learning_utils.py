from math import log, exp
from operator import add
import numpy as np
from utils import constants
from collections import defaultdict
from environment.game import Game


class Decay:
    def __init__(
        self,
        start: float,
        end: float,
        episodes: int,
        proportion_to_decay_over: float = 0.9,
    ):
        """A decay object decays a parameter from start to end over the given 
        proportion of episodes exponentially with base e

        Parameters
        ----------
        start : float
            The starting value for the parameter (episode 0)
        end : float
            The lowest value the parameter will take
        episodes : int
            The total number of episodes that training takes place over
        proportion_to_decay_over : float, optional
            The proportion of episodes that the parameter should be decayed over
            eg 0.9 means decay to end value over first 90% of training, by default 0.9
        """
        self.b = log(end / start) / (proportion_to_decay_over * episodes)
        self.episode = 0
        self.end = end

    def decay(self):
        """Gets called at the end of an episode to decay the value by one step
        """
        self.episode += 1

    def get_current_value(self) -> float:
        """Returns the current decayed value of the parameter
        """
        return max(exp(self.b * self.episode), self.end)

    def select_action(self, state: np.array, Q: defaultdict) -> str:
        if np.random.random() < self.get_current_value():
            action = np.random.choice(constants.BABY_MOVEMENTS)
        else:
            action = constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]
        return action

    def select_action_MC(self, state: np.array, Q: defaultdict) -> str:
        if np.random.random() < self.get_current_value():
            action = (np.random.choice(constants.BABY_MOVEMENTS), False)
        else:
            action = (constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])], True)
        return action

    def selection_action_double_Q(self, state, Q1, Q2):
        if np.random.random() < self.get_current_value():
            action = np.random.choice(constants.BABY_MOVEMENTS)
        else:
            action = get_action(state, Q1, Q2)
        return action

    def get_probability_selection(self, state, Q, action):
        if action == constants.BABY_MOVEMENTS[np.argmax(Q[state.tobytes()])]:
            return (1 - self.get_current_value()) + (
                1 / len(constants.BABY_MOVEMENTS)
            ) * self.get_current_value()
        else:
            return (
                self.get_current_value()
                - (1 / len(constants.BABY_MOVEMENTS)) * self.get_current_value()
            )

    def get_policy(self, Q, state):
        greedy_action_idx = np.argmax(Q[state.tobytes()])
        policy = np.repeat(
            self.get_current_value() / len(constants.BABY_MOVEMENTS),
            len(constants.BABY_MOVEMENTS),
        )
        policy[greedy_action_idx] += 1 - self.get_current_value()
        return policy


# Generating actions from the two Q-value stores
def get_action(state, Q1, Q2):
    Q_value = get_Q_value(state, Q1, Q2)
    return constants.BABY_MOVEMENTS[np.argmax(Q_value)]


# Give the sum of Q values
def get_Q_value(state, Q1, Q2):
    Q_value = [0] * 5
    if state.tobytes() in Q1.keys():
        Q_value = list(map(add, Q_value, Q1[state.tobytes()]))
    if state.tobytes() in Q2.keys():
        Q_value = list(map(add, Q_value, Q2[state.tobytes()]))
    return Q_value


# Game initialization here to save lines in the learning file
def init_game_for_learning():
    return Game(
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


def sample_action(policy):
    return constants.BABY_MOVEMENTS[
        np.random.choice(np.arange(0, len(policy)), p=policy)
    ]
