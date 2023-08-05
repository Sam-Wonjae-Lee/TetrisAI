import retro
import neat
import pickle
import numpy as np
import cv2
from PIL import Image
from typing import List


# field_end_addr = 0x04C7


def read_field(env, field_start_addr, num_rows, num_cols) -> List[List[int]]:
    """
    Represents the field as 0s and 1s.
    0 means empty tile and 1 means occupied tile.
    """
    # Get the game state from the environment RAM
    state = env.get_ram()

    # Nested list of ten 0s
    # len(field) == 20
    field = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

    # Extract the playfield data from the RAM
    field_data = state[field_start_addr: field_start_addr + (num_rows * num_cols)]

    # Populate the field array based on the data read from RAM
    for row in range(num_rows):
        for col in range(num_cols):
            cell = field_data[row * num_cols + col]
            # Initialize cells in the field, 239 from memory is an empty cell, otherwise set to 1
            field[row][col] = 0 if cell == 239 else 1

    # Return 2D array
    return field


def calculate_column_heights(field) -> List[int]:
    """
    Gives a list of the height of each column.
    Height is defined as the distance from the highest tile in each column
    to the bottom of the grid.
    """
    # Initialize the list to store the heights of each column
    column_heights = [0] * len(field[0])

    # Iterate through each column
    for col in range(len(field[0])):
        # Find the height of the column by counting from the top to the first occupied block (1)
        height = 0
        for row in range(len(field)):
            if field[row][col] == 1:
                height = len(field) - row
                break

        column_heights[col] = height

    return column_heights


def calculate_bumpiness(column_heights) -> int:
    # Initialize bumpiness to zero
    bumpiness = 0

    # Calculate the bumpiness as the sum of absolute differences in height between adjacent columns
    for i in range(len(column_heights) - 1):
        bumpiness += abs(column_heights[i] - column_heights[i + 1])

    return bumpiness


def calculate_holes(field) -> int:
    # Initialize the list to store the number of holes in each column
    column_holes = [0] * len(field[0])

    # Iterate through each column
    for col in range(len(field[0])):
        holes = 0
        block_above = False

        for row in range(len(field)):
            if field[row][col] == 1:  # Check for an occupied cell
                block_above = True
            elif field[row][col] == 0 and block_above:
                # Count the number of empty cells (holes) below the current cell in the column
                holes += 1
        column_holes[col] = holes
    return sum(column_holes)


def main():
    env = retro.make('Tetris-Nes', state='StartLv0')
    field_start_addr = 0x0400
    num_rows = 20
    num_cols = 10

    obs = env.reset()
    done = False

    # while not done:
    for i in range(10000):
        field = read_field(env, field_start_addr, num_rows, num_cols)
        print('* ' * (len(field[0])))
        for row in field:
            print(' '.join(map(str, row)))
        print('* ' * (len(field[0])))

        column_heights = calculate_column_heights(field)
        print("Column Heights:", column_heights)
        bumpiness = calculate_bumpiness(column_heights)
        print("Bumpiness:", bumpiness)
        column_holes = calculate_holes(field)
        print("Column Holes:", column_holes)

        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        env.render()

    env.close()


if __name__ == '__main__':
    main()
