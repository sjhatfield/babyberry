import sys

sys.path.append("..")

from environment.game import Game
import numpy as np
import time
import random
from typing import List, Union

SEED = 3141
MOVEMENTS = ["N", "E", "S", "W"]

random.seed(SEED)
np.random.seed(SEED)


def main():
    test_simple()
    test_initialization(5)
    test_random_movement(3, 1, 0.5, 20)
    test_render(3, 3, 20)
    test_state(size=10, num_berries=10, number_steps=10, state_sizes=[3, 5])
    test_dad(6, 3, 100, [5, 5], 0.5)
    test_dad(10, 20, 100, dad_position=-1, dad_movement_probability=0.5)
    test_smart_dad(10, 10, 50, [9, 9], 1)


def test_simple():
    """Generates a small game and manually moves the baby around the board.
    There is one berry which does not move.
    """
    game = Game(3, [0, 0], -1, 5, -5, 10, 1, [[0, 1]], [0.0])

    print(f"Check the baby exists\n{game.baby}")

    print("\nCheck the berry exists")
    for berry in game.get_berries():
        print(berry)

    print(f"\nHere is the board\n{game.get_board()}")

    print("First let's perform an illegal move Northwards")
    board, reward, done = game.step("N")
    print(f"Here is the board\n{game.get_board()}")
    print(f"And the reward experienced: {reward}")
    print(f"And whether the game is over: {done}")

    print("\nNow let's perform a legal move which does NOT eat the berry")
    board, reward, done = game.step("E")
    print(f"Here is the board\n{game.get_board()}")
    print(f"And the reward experienced: {reward}")
    print(f"And whether the game is over: {done}")

    print("\nNow we will move back to the original place and then eat the berry")
    board, reward, done = game.step("W")
    print(f"Here is the board\n{game.get_board()}")
    print(f"And the reward experienced: {reward}")
    print(f"And whether the game is over: {done}")

    print("\nNow let's perform a legal move which does NOT eat the berry")
    board, reward, done = game.step("S")
    print(f"Here is the board\n{game.get_board()}")
    print(f"And the reward experienced: {reward}")
    print(f"And whether the game is over: {done}")


def test_initialization(number: int) -> None:
    """Just shows the initialziation of the board for some random parameters

    Parameters
    ----------
    number : int
        Number of initializations to show
    """
    for _ in range(number):
        if random.random() < 0.5:
            size = random.randint(3, 10)
            baby_position = [random.randint(0, size - 1), random.randint(0, size - 1)]
            num_berries = random.randint(1, size)
        else:
            size = [random.randint(3, 10), random.randint(3, 10)]
            baby_position = [
                random.randint(0, size[0] - 1),
                random.randint(0, size[1] - 1),
            ]
            num_berries = random.randint(1, size[0])
        print(f"\n\n\nSize of the board {size}")
        print(f"Baby position: {baby_position}")
        print(f"Number of berries to be placed randomly: {num_berries}")
        game = Game(size, baby_position, 0, 0, 0, 0, num_berries)
        print(f"Here is the board:\n{game.get_board()}")
        print(game.get_baby())
        for b in game.get_berries():
            print(b)


def test_random_movement(
    size: Union[int, tuple], num_berries: int, delay_seconds: int, number_steps: int
) -> None:
    """Displays the baby taking random movements around the board with the experienced reward

    Parameters
    ----------
    size : Union[int, tuple]
        Size of the board 
    num_berries : int
        Number of berries to be placed randomly on the board
    delay_seconds : int
        Delay in seconds to view the board inbetween moves
    number_steps : int
        Number of actions to be taken by the baby
    """
    game = Game(
        size,
        [0, 0],
        -1,
        5,
        -5,
        10,
        num_berries,
        berry_movement_probabilities=[0.5] * num_berries,
    )
    print(f"Starting board:\n{game.get_board()}")
    done = False
    i = 1
    while not done and i < number_steps:
        print(f"Action {i}")
        time.sleep(delay_seconds)
        _, reward, done = game.step(random.choice(MOVEMENTS))
        print(f"Board:\n{game.get_board()}")
        print(f"Reward: {reward}")
        i += 1


def test_render(size: Union[int, tuple], num_berries: int, number_steps: int) -> None:
    """Shows some random movement rendered by opencv

    Parameters
    ----------
    size : Union[int, tuple]
        Size of the board
    num_berries : int
        Number of berries placed on the board
    number_steps : int
        Number of steps to be shown
    """
    game = Game(
        size,
        [0, 0],
        -1,
        5,
        -5,
        10,
        num_berries,
        berry_movement_probabilities=[0.5] * num_berries,
    )
    done = False
    i = 1
    while not done and i < number_steps:
        _, reward, done = game.step(random.choice(MOVEMENTS), True)
        print(f"reward: {reward}")
        print(f"number of berries: {len(game.berries)}")
        i += 1


def test_state(
    size: Union[int, tuple],
    num_berries: int,
    number_steps: int,
    state_sizes: List[int] = [3, 5],
) -> None:
    """Shows both the full board and the state of the local area of the board around the baby which the 
    learners will see

    Parameters
    ----------
    size : Union[int, tuple]
        Size of the board
    num_berries : int
        Number of berries placed on the board
    number_steps : int
        Number of actions taken by the baby to be shown
    state_sizes : List[int], optional
        Square size of the state to be shown with the baby in the center, by default [3, 5]
    """
    for state_size in state_sizes:
        game = Game(
            size,
            [0, 0],
            -1,
            5,
            -5,
            10,
            num_berries,
            berry_movement_probabilities=[0.5] * num_berries,
            state_size=state_size,
        )
        done = False
        i = 1
        print(f"Beginning full board\n{game.get_state(full=True)}")
        print(f"And the state\n{game.get_state(state_size)}")
        while not done and i < number_steps:
            action = random.choice(MOVEMENTS)
            print(f"Action taken {action}")
            state, reward, done = game.step(action)
            print(f"Full board\n{game.get_state(full=True)}")
            print(f"The state\n{game.get_state(state_size)}")
            i += 1


def test_dad(
    size: Union[int, tuple],
    num_berries: int,
    number_steps: int,
    dad_position: tuple = None,
    dad_movement_probability=None,
):
    game = Game(
        size,
        [0, 0],
        -1,
        5,
        -5,
        10,
        num_berries,
        berry_movement_probabilities=[0.5] * num_berries,
        dad_initial_position=dad_position,
        dad_movement_probability=dad_movement_probability,
    )
    done = False
    i = 1
    while not done and i < number_steps:
        _, reward, done = game.step(random.choice(MOVEMENTS), True)
        print(game.get_board())
        print(f"reward: {reward}")
        print(f"number of berries: {len(game.berries)}")
        i += 1


def test_smart_dad(
    size: Union[int, tuple],
    num_berries: int,
    number_steps: int,
    dad_position: tuple = None,
    dad_movement_probability=None,
):
    game = Game(
        size,
        [0, 0],
        -1,
        5,
        -5,
        10,
        num_berries,
        berry_movement_probabilities=[0.5] * num_berries,
        dad_initial_position=dad_position,
        dad_movement_probability=dad_movement_probability,
        dad_dumb=False,
    )
    done = False
    i = 1
    while not done and i < number_steps:
        action = random.choice(MOVEMENTS)
        print(f"baby action chosen: {action}")
        _, reward, done = game.step(action, True)
        print(f"board after action performed\n{game.get_board()}")
        print(f"reward: {reward}")
        print(f"number of berries: {len(game.berries)}\n\n\n")
        i += 1


if __name__ == "__main__":
    main()
