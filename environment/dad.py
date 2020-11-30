import sys

sys.argv.append("..")

from utils import constants

from environment.berry import Berry
from typing import Union
import numpy as np


class Dad(Berry):
    def __init__(
        self,
        board_size: Union[int, tuple],
        movement_probability: float = None,
        initial_position: list = None,
        dumb: bool = True,
    ):
        """Creates a dad, possibly with a movement probability and initial position
        The dad may be dumb and moves randomly or not dumb (smart) which means he moves
        towards the baby

        Parameters
        ----------
        board_size : Union[int, tuple]
            Board size may be an int meaning square board or tuple
        movement_probability : float, optional
            Probability that the dad moves at each step, by default None
        initial_position : list, optional
            Where the dad starts, by default None
        dumb : bool, optional
            Whether the dad is dumb or not (smart), by default True
        """
        Berry.__init__(self, board_size, movement_probability, initial_position)
        self.dumb = dumb

    def action(self, direction: str, baby_position: tuple) -> None:
        """Takes the given action according to the movement probability
        But first checks if the baby is adjacent to dad and moves dad onto
        the baby

        Parameters
        ----------
        direction : str
            One of the allowed movements
        baby_position : tuple
            Where the baby is located
        """
        assert direction in constants.BABY_MOVEMENTS
        # First move to pick up baby is they are adjacent
        if baby_position[0] == self.position[0]:
            if baby_position[1] == self.position[1] - 1:
                self.position[1] -= 1
                return
            elif baby_position[1] == self.position[1] + 1:
                self.position[1] += 1
                return
        elif baby_position[1] == self.position[1]:
            if baby_position[0] == self.position[0] - 1:
                self.position[0] -= 1
                return
            elif baby_position[0] == self.position[0] + 1:
                self.position[0] += 1
                return

        # not adjacent
        if np.random.random() < self.movement_probability:
            if self.dumb:
                if direction == "N":
                    if self.position[0] != 0:
                        self.position[0] -= 1
                elif direction == "E":
                    if self.position[1] != self.board_dimensions[1] - 1:
                        self.position[1] += 1
                elif direction == "S":
                    if self.position[0] != self.board_dimensions[0] - 1:
                        self.position[0] += 1
                elif direction == "W":
                    if self.position[1] != 0:
                        self.position[1] -= 1
            else:
                # Find out whether the baby is further away row-wise
                # or column-wise to decide movement
                dad_pos = self.position.copy()
                row_diff = baby_position[0] - dad_pos[0]
                col_diff = baby_position[1] - dad_pos[1]
                # Move in the direction with greatest difference
                if abs(row_diff) > abs(col_diff):
                    if row_diff > 0:
                        self.position[0] += 1
                    else:
                        self.position[0] -= 1
                    return
                elif abs(row_diff) < abs(col_diff):
                    if col_diff > 0:
                        self.position[1] += 1
                    else:
                        self.position[1] -= 1
                    return
                elif abs(row_diff) == abs(col_diff):
                    if np.random.random() < 0.5:
                        if row_diff > 0:
                            self.position[0] += 1
                        else:
                            self.position[0] -= 1
                    else:
                        if col_diff > 0:
                            self.position[1] += 1
                        else:
                            self.position[1] -= 1
