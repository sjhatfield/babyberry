import sys

sys.argv.append("..")

from utils import constants
import numpy as np
from typing import Union


class Berry:
    """
    Berries are to be eaten by the baby and are randomly spawned 
    in the environment. They roll around randomly in one of the 
    four compass directions.
    """

    def __init__(
        self,
        board_size: Union[int, tuple],
        movement_probability: float = None,
        initial_position: list = None,
    ):
        """Initializes a berry

        Parameters
        ----------
        board_size : Union[int, tuple]
            This is the size of the board as an integer if it is a square board
            or tuple if rectangular
        movement_probability : float, optional
            This is the probability that the berry will move one of the four directions per timestep, by default None
        initial_position : list, optional
            This is the 2 coordinate starting position of a berry, by default None
        """
        if type(board_size) == int:
            self.board_dimensions = (board_size, board_size)
        else:
            self.board_dimensions = board_size

        if movement_probability:
            assert (
                0 <= movement_probability <= 1
            ), "Movement probability needs to be a float between 0 and 1"
            self.movement_probability = movement_probability
        else:
            self.movement_probability = constants.DEFAULT_MOVEMENT_PROBABILITY

        if initial_position:
            assert (
                len(initial_position) == 2
            ), "Position must be a list of length 2 containing x and y coordinates where top left of the board is [0,0]"
            assert (
                0 <= initial_position[0] < self.board_dimensions[0]
            ), "Invalid initial x position"
            assert (
                0 <= initial_position[1] < self.board_dimensions[1]
            ), "invalid initial y position"
            self.position = initial_position.copy()
        else:
            self.position = [
                np.random.randint(0, self.board_dimensions[0] - 1),
                np.random.randint(0, self.board_dimensions[1] - 1),
            ]

    def action(self, direction: str) -> None:
        """The berry randomly moves in one of the four compass directions with
        probability given upon initialization

        Parameters
        ----------
        direction : str
            One of N, E, S, W
        """
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

    def get_position(self):
        return self.position.copy()

    def __str__(self):
        position = self.get_position()
        return f"Berry at position ({position[0]}, {position[1]}), (row, col)"
