import sys

sys.path.append("..")

import numpy as np
import cv2
from typing import Union, List
from utils import constants
from environment.baby import Baby
from environment.berry import Berry
from environment.dad import Dad


class Game:
    """
    The game contains the board, one baby and some berries, The baby tries to 
    eat all the berries whilst the berries roll around randomly.
    """

    def __init__(
        self,
        board_size: Union[int, tuple],
        baby_initial_position: list,
        move_reward: float,
        eat_reward: float,
        illegal_move_reward: float,
        complete_reward: float,
        num_berries: int = None,
        berry_positions: List[list] = [None],
        berry_movement_probabilities: List[float] = [None],
        dad_initial_position: Union[List[int], int] = None,
        dad_movement_probability: float = 0.5,
        dad_dumb: bool = True,
        game_over_reward: float = -20,
        state_size: int = 3,
    ):
        """Initializes the game with a baby and some berries. Positions for some or all of the 
        berries can be given. If they are not given then they take random positions. 
        Probabilities that each of the berries will move at each time step are given also.
        If no probability is given then 50% probability of movement is used. Finally, the number of
        berries on the board can be given. If not, then a radnom number of berries will be spawned 
        between 1 and the square root of the number of squares on the board.

        Parameters
        ----------
        board_size : Union[int, tuple]
            If integer the board is a square, if tuple rectangular
        baby_position : list
            Two coordinate position of a baby
        move_reward : float
            The reward experienced (most likely negative) for moving
        eat_reward : float
            The reward (most likely positive) for eating a berry
        illegal_move_reward : float
            The reward (negative) for trying to move off the board
        complete_reward : float
            The reward (positive) for completing the game
        num_berries : int, optional
            The number of berries in the game, by default None
        berry_positions : List[list], optional
            The positions of the berries, fewer positions than berries can be given, by default [None]
        berry_movement_probabilities : List[float], optional
            The berry movement probabilities as floats, fewer can be given than the berries, by default [None]
        dad_position: List[int], default = None
            If not none means there is a dad on the board to pick up the baby, gives the starting position of the dad
        dad_movement_probability : float
            Probility that if the dad cannot pick up the baby he will take a random movement
        state_size : int, by default 3
            This is the size of the window around the baby that is returned as the state
        """
        if type(board_size) == int:
            assert board_size > 0, "Board size must be positive integer"
            assert (
                num_berries < board_size ** 2 - 1
            ), "That many berries cannot fit on the board"
            self.board_dimensions = (board_size, board_size)
        else:
            assert (
                board_size[0] > 0 and board_size[1] > 0
            ), "Board dimensions must be positive integers"
            assert (
                num_berries < board_size[0] * board_size[1] - 1
            ), "That many berries cannot fit on the board"
            self.board_dimensions = board_size
        assert state_size >= 3, "Size of square grid for state must be at least 3 by 3"
        assert (
            state_size % 2 == 1
        ), "Size must be an odd integer greater than or equal 3, so that the baby is positioned in the center"
        assert (
            baby_initial_position != dad_initial_position
        ), "Baby and dad cannot start in the same space at the beginning"
        if dad_initial_position:
            assert self.board_dimensions[0] * self.board_dimensions[1] > 9, (
                "If the game is to have a dad the board needs to be bigger than 3"
                " by 3 so that the dad may be positioned not to immediately pick up the baby"
            )

        self.baby_initial_position = baby_initial_position.copy()
        self.move_reward = move_reward
        self.illegal_move_reward = illegal_move_reward
        self.eat_reward = eat_reward
        self.complete_reward = complete_reward
        self.game_over_reward = game_over_reward
        if num_berries:
            self.num_berries = num_berries
        else:
            self.num_berries = np.random.randint(
                1, int((self.board_dimensions[0] * self.board_dimensions[1]) ** 0.5)
            )
        self.berries = []
        self.berry_movement_probabilities = berry_movement_probabilities
        self.berry_positions = berry_positions
        self.state_size = state_size

        # Pad the lists containing berry movement probabilities and positions
        # with Nones if there are missing entries for last berries
        if len(self.berry_movement_probabilities) < num_berries:
            self.berry_movement_probabilities += [
                constants.DEFAULT_MOVEMENT_PROBABILITY
                for _ in range(num_berries - len(self.berry_movement_probabilities))
            ]
        if len(self.berry_positions) < num_berries:
            self.berry_positions += [
                None for _ in range(num_berries - len(self.berry_positions))
            ]

        self.dad = None
        if dad_initial_position != None:
            if type(dad_initial_position) != list:
                self.dad_initial_position = [
                    np.random.randint(0, self.board_dimensions[0]),
                    np.random.randint(0, self.board_dimensions[1]),
                ]
                # If we are randomly choosing a position for dad make sure he is not
                # within the 3x3 grid surrounding the baby so he can't immediately
                # pick up the baby
                while (
                    self.dad_initial_position[0]
                    in [
                        self.baby_initial_position[0] - 1,
                        self.baby_initial_position[0],
                        self.baby_initial_position[0] + 1,
                    ]
                ) or (
                    self.dad_initial_position[1]
                    in [
                        self.baby_initial_position[1] - 1,
                        self.baby_initial_position[1],
                        self.baby_initial_position[1] + 1,
                    ]
                ):
                    self.dad_initial_position = [
                        np.random.randint(0, self.board_dimensions[0]),
                        np.random.randint(0, self.board_dimensions[1]),
                    ]
            else:
                self.dad_initial_position = dad_initial_position
        else:
            self.dad_initial_position = None
        self.dad_movement_probability = dad_movement_probability
        self.dad_dumb = dad_dumb

        self.reset()

    def reset(self):
        """Resets the board to its original configuration. For berries with no 
        position a random one is used so these will change over different episodes
        """
        self.baby = Baby(self.board_dimensions, self.baby_initial_position)
        if self.dad_initial_position != None:
            self.dad = Dad(
                self.board_dimensions,
                self.dad_movement_probability,
                self.dad_initial_position,
                self.dad_dumb,
            )
        self.berries = []

        for i in range(self.num_berries):
            if self.berry_positions[i]:
                new_berry = Berry(
                    self.board_dimensions,
                    self.berry_movement_probabilities[i],
                    [self.berry_positions[i][0], self.berry_positions[i][1]],
                )
            else:
                new_berry = Berry(
                    self.board_dimensions, self.berry_movement_probabilities[i]
                )
            while self.check_position_clash(new_berry):
                if self.berry_positions[i]:
                    new_berry = Berry(
                        self.dimensions,
                        self.berry_movement_probabilies[i],
                        [self.berry_positions[i][0], self.berry_positions[i][1]],
                    )
                else:
                    new_berry = Berry(
                        self.board_dimensions, self.berry_movement_probabilities[i]
                    )
            self.berries.append(new_berry)

        self.make_board()

        return self.get_state(), 0, False

    def check_position_clash(self, berry: Berry, berries: List[Berry] = None) -> bool:
        """For a berry that is going to be placed on the board this 
        checks whether it is in the same place as any already placed 
        berry or the baby. A list of berries to be checked can be passed. 
        If it is not then all berries on the board are checked.

        Parameters
        ----------
        berry : Berry
            The berry to be checked against the others for a clash
        berries : List[Berry], optional
            The list of berries to be checked, if none then all the berries in the game are checked, by default None

        Returns
        -------
        bool
            True if clash, False if no clash
        """
        if berries == None:
            berries = self.berries
        for b in berries:
            if b.get_position() == berry.get_position():
                return True
        if self.baby.get_position() == berry.get_position():
            return True
        if self.dad:
            if self.dad.get_position() == berry.get_position():
                return True
        return False

    def move_berries(self):
        """Perform movement of all the berries on the board. One by one
        their possible legal moves are found which do not result in
        sharing a square with the baby or other berries, then a 
        random movement is selected with the appropriate probability
        """
        for i in range(len(self.berries)):
            # Find all the legal moves that this berry can take.
            # A legal move results in them occupying an empty space
            legal_moves = []
            for move in constants.BERRY_MOVEMENTS:
                dummy_berry = Berry(
                    self.board_dimensions, 0, self.berries[i].get_position().copy()
                )
                dummy_berry.action(move)
                # Check whether the move results in a clash comparing to all berries on the board
                if not self.check_position_clash(dummy_berry, self.berries):
                    legal_moves.append(move)
            # It is possible that there are no legal moves for a berry if it is surrounded
            # In that case do not move
            if len(legal_moves) != 0:
                if np.random.random() < self.berry_movement_probabilities[i]:
                    move = np.random.choice(legal_moves)
                    self.berries[i].action(move)

            self.make_board()

    def step(self, action, render: bool = False):
        """
        This function performs the baby action and updates the board accordingly.
        It returns the new state of the board, the reward received by the baby and
        whether this action has resulted in the game being finished (all berries eaten)
        """
        # First check if dad is on baby
        if self.dad != None:
            if self.dad.get_position() == self.baby.get_position():
                self.make_board()
                return (
                    self.get_state(),
                    self.game_over_reward,
                    True,
                )
        self.move_berries()
        # First move the dad. He will pick up the baby if adjacent
        if self.dad:
            move = np.random.choice(constants.BERRY_MOVEMENTS)
            self.dad.action(move, self.baby.get_position())
            # Check if the dad and baby now occupy the same space meaning the
            # game is over and the baby failed to eat all the berries
            if self.dad.get_position() == self.baby.get_position():
                self.make_board()
                return (
                    self.get_state(),
                    self.game_over_reward,
                    True,
                )
        self.make_board()

        # Now move the baby
        move_legality = self.baby.action(action)
        if not move_legality:
            return (
                self.get_state(),
                self.illegal_move_reward,
                False,
            )

        reward = self.move_reward

        # Find out if any berries now occupy the same space as the baby
        # meaning they have been eaten. There should be either 0 or 1
        # berry in the same space as the baby
        eaten_idx = []
        for i, b in enumerate(self.berries):
            if self.baby.get_position() == b.get_position():
                reward += self.eat_reward
                eaten_idx.append(i)

        assert len(eaten_idx) in [
            0,
            1,
        ], "Something is wrong with the code as more than 1 berry has been eaten by the baby!"
        if len(eaten_idx) > 0:
            del self.berries[eaten_idx[0]]

        self.make_board()

        if render:
            self.render()

        # Now we check if the game is over
        if len(self.berries) == 0:
            reward += self.complete_reward
            return self.get_state(), reward, True

        # The game is not over
        return self.get_state(), reward, False

    def make_board(self):
        """
        Generates the board given the current situation of baby and berries
        """
        self.board = np.full(
            shape=self.board_dimensions,
            fill_value=constants.CHARACTER_KEY["tile"],
            dtype=np.int8,
        )
        baby_pos = self.baby.get_position()
        self.board[(baby_pos[0], baby_pos[1])] = constants.CHARACTER_KEY["baby"]
        i = 1
        for b in self.berries:
            berry_position = b.get_position()
            self.board[(berry_position[0], berry_position[1])] = i
            i += 1
        if self.dad:
            dad_pos = self.dad.get_position()
            self.board[(dad_pos[0], dad_pos[1])] = constants.CHARACTER_KEY["dad"]

    def get_board(self):
        return self.board.copy()

    def get_board_image(self):
        dims = self.board_dimensions
        image = np.full(
            shape=(dims[0], dims[1], 3),
            fill_value=constants.COLORS["tile"],
            dtype=np.uint8,
        )
        baby_pos = self.baby.get_position()
        image[(baby_pos[0], baby_pos[1])] = constants.COLORS["baby"]
        for b in self.berries:
            berry_position = b.get_position().copy()
            image[(berry_position[0], berry_position[1])] = constants.COLORS["berry"]
        if self.dad:
            dad_pos = self.dad.get_position().copy()
            image[(dad_pos[0], dad_pos[1])] = constants.COLORS["dad"]
        return image

    def render(self):
        img = cv2.resize(
            self.get_board_image()[..., ::-1].copy() / 255.0,
            (300, 300),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Environment Render", img)
        cv2.waitKey(constants.WAIT_MS)

    def get_berries(self):
        return self.berries

    def get_baby(self):
        return self.baby

    def get_state_size(self):
        return self.state_size

    def get_state(self, full: bool = False) -> np.array:
        """Returns a square numpy array around the baby to represent current state

        Parameters
        ----------
        full : bool, default = False
            If true we return the full board with edge padding rather than the local
            neighborhood of the baby

        Returns
        -------
        np.array
            State around the baby as a numpy square array of integers
        """
        size = self.get_state_size()

        state = np.zeros(shape=(size, size), dtype=np.int8)
        padding = int((size - 1) / 2)
        board_size = self.board.shape

        padded_board = np.full(
            shape=(board_size[0] + 2 * padding, board_size[1] + 2 * padding),
            fill_value=constants.CHARACTER_KEY["tile"],
            dtype=np.int8,
        )
        padded_board[:padding, :] = constants.CHARACTER_KEY["edge"]
        padded_board[:, :padding] = constants.CHARACTER_KEY["edge"]
        padded_board[-padding:, :] = constants.CHARACTER_KEY["edge"]
        padded_board[:, -padding:] = constants.CHARACTER_KEY["edge"]
        padded_board[padding:-padding, padding:-padding] = self.get_board().copy()

        baby_pos = self.baby.get_position()
        baby_pos_in_padded = tuple(map(sum, zip(baby_pos, (padding, padding))))
        padded_board[padded_board >= 1] = 1

        if full:
            return padded_board
        return padded_board[
            baby_pos_in_padded[0] - padding : baby_pos_in_padded[0] + padding + 1,
            baby_pos_in_padded[1] - padding : baby_pos_in_padded[1] + padding + 1,
        ]

