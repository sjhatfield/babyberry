import sys

sys.argv.append("..")

from utils import constants
import numpy as np


class Baby:
    """
    The baby is the player who is controlled by the RL agent.
    The baby moves around the game board trying to eat the 
    berries. The game is over once all of the berries are eaten
    """

    def __init__(self, board_dimensions: tuple, initial_position: list = None):
        """
        Initializes the baby character in a board. If intial coordinates
        are given then they are placed there. Otherwise placement is random.
        Position should be given as (row, column) like numpy
        """
        self.board_dimensions = board_dimensions
        if initial_position:
            assert type(initial_position) == list, "Position must be length 2 list"
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
                np.random.randint(0, board_dimensions[0] - 1),
                np.random.randint(0, board_dimensions[1] - 1),
            ]

    def action(self, direction: str) -> bool:
        """
        Executes a one step movement in either of the 4 compass
        directions. Returns True is the move was legal and returns
        False if the move was illegal. Coordinate are (row, col)
        """
        direction = direction[0].upper()
        assert (
            direction in constants.BABY_MOVEMENTS
        ), f"Movement must be one of {constants.BABY_MOVEMENTS}"
        if direction == "R":
            legal_moves = []
            if self.position[0] != 0:
                legal_moves.append("N")
            if self.position[0] != self.board_dimensions[0] - 1:
                legal_moves.append("S")
            if self.position[1] != 0:
                legal_moves.append("W")
            if self.position[1] != self.board_dimensions[1] - 1:
                legal_moves.append("E")
            direction = np.random.choice(legal_moves)
        if direction == "N":
            if self.position[0] != 0:
                self.position[0] -= 1
                return True
            else:
                return False
        elif direction == "E":
            if self.position[1] != self.board_dimensions[1] - 1:
                self.position[1] += 1
                return True
            else:
                return False
        elif direction == "S":
            if self.position[0] != self.board_dimensions[0] - 1:
                self.position[0] += 1
                return True
            else:
                return False
        elif direction == "W":
            if self.position[1] != 0:
                self.position[1] -= 1
                return True
            else:
                return False
        return False

    def get_position(self):
        """Returns the position on the baby as a tuple of ints

        Returns
        -------
        tuple(int)
        """
        return self.position.copy()

    def get_board_dimensions(self) -> list:
        """Returns the board dimensions

        Returns
        -------
        list(int)
        """
        return self.board_dimensions.copy()

    def __str__(self):
        """For printing the position of a baby

        Returns
        -------
        str
        """
        position = self.get_position()
        return f"Baby at position ({position[0]}, {position[1]}) (row, col)"
