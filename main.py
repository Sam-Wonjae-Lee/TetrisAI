import retro
import neat
import pickle
import numpy as np
import cv2
from PIL import Image


# field_end_addr = 0x04C7


def read_field(env, field_start_addr, num_rows, num_cols):
    # Get the game state from the environment RAM
    state = env.get_ram()

    field = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

    # Extract the playfield data from the RAM
    field_data = state[field_start_addr: field_start_addr + (num_rows * num_cols)]

    # Populate the field array based on the data read from RAM
    for row in range(num_rows):
        for col in range(num_cols):
            cell = field_data[row * num_cols + col]
            # Initialize cells in the field, 239 from memory is an empty cell, otherwise set to 1
            field[row][col] = 0 if cell == 239 else 1

    return field


def main():
    env = retro.make('Tetris-Nes', state='StartLv0')
    field_start_addr = 0x0400
    num_rows = 20
    num_cols = 10

    obs = env.reset()
    done = False

    while not done:
        field = read_field(env, field_start_addr, num_rows, num_cols)
        for row in field:
            print(' '.join(map(str, row)))

        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        env.render()

    env.close()


if __name__ == '__main__':
    main()
