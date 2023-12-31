""" Tetris AI
Created by Wonjae Lee

Description:
This file contains all code to the run the program, including running Tetris and the program playing it.

Instructions:
1. Python must be run on version 3.7
2. Install all requirements from requirements.txt
3. Run main.py file

"""


import random
import retro
import pickle
import cv2
from typing import List, Dict, Tuple
import neat

import copy
import numpy as np

# import pieces
# from PIL import Image
# import matplotlib.pyplot as plt

# field_end_addr = 0x04C7
# current_piece_addr = 0x0062
# next_piece_addr = 0x00BF
# "buttons": ["B", null, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]


""" For info on each tetromino piece """

# current_piece_addr = 0x0062
# next_piece_addr = 0x00BF

""" Value of each piece using current_piece_addr and next_piece_addr 

    Decimal Value (Hexadecimal Value): Piece Type

    0 (00): T up
    1 (01): T right
    2 (02): T down (spawn)
    3 (03): T left

    4 (04): J left
    5 (05): J up
    6 (06): J right
    7 (07): J down (spawn)

    8 (08): Z horizontal (spawn)
    9 (09): Z vertical 

    10 (0A): O (spawn)

    11 (0B): S horizontal (spawn)
    12 (0C): S vertical

    13 (0D): L right
    14 (0E): L down (spawn)
    15 (0F): L left
    16 (10): L up

    17 (11): I vertical
    18 (12): I horizontal (spawn)

"""

"""
Dictionary of each tetromino spawn value paired with possible rotations value
"""
possible_rotations = {
    0: {0, 1, 2, 3},
    1: {0, 1, 2, 3},
    2: {0, 1, 2, 3},  # T
    3: {0, 1, 2, 3},

    4: {4, 5, 6, 7},
    5: {4, 5, 6, 7},
    6: {4, 5, 6, 7},
    7: {4, 5, 6, 7},  # J

    8: {8, 9},  # Z
    9: {8, 9},

    10: {10, 19},  # O

    11: {11, 12},  # S
    12: {11, 12},

    13: {13, 14, 15, 16},
    14: {13, 14, 15, 16},  # L
    15: {13, 14, 15, 16},
    16: {13, 14, 15, 16},

    17: {17, 18},
    18: {17, 18},  # I

    19: {10, 19}  # O
}

pieces_shapes = {

    0: [[0, 1, 0],  # T up
        [1, 1, 1]],

    1: [[1, 0],  # T right
        [1, 1],
        [1, 0]],

    2: [[1, 1, 1],  # T down
        [0, 1, 0]],

    3: [[0, 1],  # T left
        [1, 1],
        [0, 1]],

    4: [[0, 1],  # J left
        [0, 1],
        [1, 1]],

    5: [[1, 0, 0],  # J up
        [1, 1, 1]],

    6: [[1, 1],  # J right
        [1, 0],
        [1, 0]],

    7: [[1, 1, 1],  # J down
        [0, 0, 1]],

    8: [[1, 1, 0],  # Z horizontal
        [0, 1, 1]],

    9: [[0, 1],  # Z vertical
        [1, 1],
        [1, 0]],

    10: [[1, 1],  # O
         [1, 1]],

    11: [[0, 1, 1],  # S horizontal
         [1, 1, 0]],

    12: [[1, 0],  # S vertical1//vertical1
         [1, 1],
         [0, 1]],

    13: [[1, 0],  # L right
         [1, 0],
         [1, 1]],

    14: [[1, 1, 1],  # L down
         [1, 0, 0]],

    15: [[1, 1],  # L left
         [0, 1],
         [0, 1]],

    16: [[0, 0, 1],  # L up
         [1, 1, 1]],

    17: [[1],  # I vertical
         [1],
         [1],
         [1]],

    18: [[1, 1, 1, 1]],  # I horizontal

    19: [[1, 1],  # O
         [1, 1]]
}

# Start drop column of the left side of a piece
piece_column = {
    0: 5,
    1: 6,
    2: 5,
    3: 5,
    4: 5,
    5: 5,
    6: 6,
    7: 5,
    8: 5,
    9: 6,
    10: 5,
    11: 5,
    12: 6,
    13: 6,
    14: 5,
    15: 5,
    16: 5,
    17: 6,
    18: 4,
    19: 5
}

# The pivot point of rotation for each piece
piece_pivot = {
    0: (1, 1),
    1: (1, 0),
    2: (0, 1),
    3: (1, 1),
    4: (1, 1),
    5: (1, 1),
    6: (1, 0),
    7: (0, 1),
    8: (0, 1),
    9: (1, 0),
    10: (0, 1),
    11: (0, 1),
    12: (1, 0),
    13: (1, 0),
    14: (0, 1),
    15: (1, 1),
    16: (1, 1),
    17: (2, 0),
    18: (0, 2),
    19: (0, 1)

    # 0: pieces_shapes[0][1][1],
    # 1: pieces_shapes[1][1][0],
    # 2: pieces_shapes[2][0][1],
    # 3: pieces_shapes[3][1][1],
    # 4: pieces_shapes[4][1][1],
    # 5: pieces_shapes[5][1][1],
    # 6: pieces_shapes[6][1][0],
    # 7: pieces_shapes[7][0][1],
    # 8: pieces_shapes[8][0][1],
    # 9: pieces_shapes[9][1][0],
    # 10: pieces_shapes[10][0][1],
    # 11: pieces_shapes[11][0][1],
    # 12: pieces_shapes[12][1][0],
    # 13: pieces_shapes[13][1][0],
    # 14: pieces_shapes[14][0][1],
    # 15: pieces_shapes[15][1][1],
    # 16: pieces_shapes[16][1][1],
    # 17: pieces_shapes[17][2][0],
    # 18: pieces_shapes[18][0][2]

}


def get_possible_fields(playfield: List[List[int]], current_piece_val: int, num_cols=10):
    """
    Gets all possible fields based on the current playfield and current_piece_val.

    playfield is the current board state in Tetris (0s are empty cells and 1s are non-empty cells).
    current_piece_val represents the various tetrominos in their default state (when they spawn/falling). Info about
        the different tetrominos is at the top.
    num_cols is the number of columns in playfield.
        NOTE: Could calculate num_cols within the method based on playfield.

    Preconditions:
        - current_piece_val in [2, 7, 8, 10, 11, 14, 18]
        - playfield is a valid Tetris board with 0s and 1s

    >>> initial_field = [[0 for _ in range(10)] for _ in range(20)]
    >>> get_possible_fields(initial_field, 18, 10)
    ...
    """
    column_heights = calculate_column_heights(playfield)
    possible_fields_list = []
    current_piece_rotations = possible_rotations[current_piece_val]

    for current_piece_rotation in current_piece_rotations:
        piece = pieces_shapes[current_piece_rotation]

        for col in range(1, num_cols - len(piece[0]) + 2):
            # Create copies of playfield and column_heights for each iteration

            updated_playfield = drop_piece(playfield, column_heights, piece, col)

            # Calculate attributes
            col_heights = calculate_column_heights(updated_playfield)

            updated_field_data = {
                'field': updated_playfield,
                'col_heights': col_heights,
                'best_col': col,
                'best_piece': current_piece_val,
                'best_rotation': current_piece_rotation
            }

            possible_fields_list.append(updated_field_data)

    return possible_fields_list


def merge_tables(field: List[List[int]], piece_shape: List[List[int]], col: int, row: int) -> List[List[int]]:
    """
    Merges the field with the piece_shape.

    piece_shape is from pieces_shapes.
    col and row represent which column and row the piece (from left side) will be placed in the field.
    field will be mutated.

    Preconditions:
        - 1 <= col <= 10
        - 1 <= row <= 20
        - field is a valid Tetris board with 0s and 1s
        - piece_shape in pieces_shapes.values()

    >>> small_field = [[0 for _ in range(4)] for _ in range(4)]
    >>> merge_tables(small_field, pieces_shapes[2], 2, 3)
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]]
    """
    for i in range(len(piece_shape)):  # Iterate through rows
        for j in range(len(piece_shape[0])):  # Iterate through columns of current row
            if piece_shape[i][j] == 1:
                field[row + i - 1][col + j - 1] = piece_shape[i][j]
    return field


def tables_equal(t1, t2):
    if len(t1) != len(t2):
        return False
    for i in range(len(t1)):
        if t1[i] != t2[i]:
            return False
    return True


def add_tables(t1, t2):
    result = []
    for i in range(min(len(t1), len(t2))):
        result.append(t1[i] + t2[i])
    return result


def get_piece_bottom(piece_shape: List[List[int]]):
    """
    Returns the bottom row and offset of the tetromino piece.
    offset is defined as how many empty rows are there below the piece's bottom row.

    piece_shape is from pieces_shapes.

    Preconditions:
        - piece_shape in pieces_shapes.values()

    >>> get_piece_bottom(pieces_shapes[2])
    ([0, 1, 0], 1)
    """
    row = piece_shape[-1].copy()
    bottom = row.copy()
    offset = 0
    rows_equal = True

    for i in range(len(piece_shape) - 2, -1, -1):
        if tables_equal(piece_shape[i], row) and rows_equal:
            bottom = add_tables(bottom, piece_shape[i])
        else:
            offset += 1
            rows_equal = False

    return bottom, offset


def drop_piece(field: List[List[int]], col_heights: List[int], piece_shape: List[List[int]], col: int) -> List[
    List[int]]:
    """
    Returns a board state where the piece is dropped in the chosen column (col). This function detects for collision.

    col_heights is a list of the height of each column based on field. It is calculated beforehand
        in get_possible_fields.
    col is to decide where the piece (from left side) should be dropped.

    Preconditions:
        - field is a valid Tetris board with 0s and 1s
        - piece_shape in pieces_shapes.values()
        - 1 <= col <= 10
    """
    bottoms, offset = get_piece_bottom(piece_shape)
    # print(offset)

    # sums is a list used to calculate the total height at which the dropped piece will land on the playfield for each
    # possible column.
    sums = []
    # for i in range(col, col + len(piece_shape[0])):
    for i in range(col, col + len(bottoms)):
        # col_heights has a max index of 9
        sums.append(col_heights[i - 1] + bottoms[i - col])

    row = max(sums) + offset
    # print(sums)
    # print(row)
    # To prevent mutation of field
    field_copy = [row[:] for row in field]
    updated_playfield = merge_tables(field_copy, piece_shape, col, 21 - row)

    return updated_playfield


# ======================================================================================================


def read_field(env: retro.retro_env.RetroEnv, field_start_addr=0x0400, num_rows=20, num_cols=10) -> List[List[int]]:
    """
    Reads the current state (env) of the board by extracting the RAM data to represents the board (field) as 0s and 1s.
    0 means empty tile and 1 means occupied tile. The field start address is 0x0400.
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


def update_field_with_piece(field: List[List[int]], current_piece_val: int, piece_position: Tuple[int, int]):
    """
    Adjusts the piece's pivot position based on ram to piece position from the top left of the piece.
    Then the current piece will be combined with the field.
    """
    piece_shape = pieces_shapes[current_piece_val]
    pivot_x, pivot_y = piece_pivot[current_piece_val]
    field_with_piece = [row[:] for row in field]

    # Position from ram to position based on left side of piece
    adjusted_x = piece_position[0] - pivot_y
    adjusted_y = piece_position[1] - pivot_x

    for i in range(len(piece_shape)):
        for j in range(len(piece_shape[0])):
            if piece_shape[i][j] == 1:
                row = adjusted_y + i
                col = adjusted_x + j
                field_with_piece[row][col] = 1
    return field_with_piece


def calculate_column_heights(field: List[List[int]]) -> List[int]:
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


def calculate_bumpiness(column_heights: List[int]) -> int:
    """
    Returns the bumpiness based on the board's column heights.
    Bumpiness calculates how smooth the top of the board is.

    0 1 0 0
    0 1 1 1 -> bumpiness = |1 - 3| + |3 - 2| + |2 - 2| = 3
    1 1 1 1
    """
    # Initialize bumpiness to zero
    bumpiness = 0

    # Calculate the bumpiness as the sum of absolute differences in height between adjacent columns
    for i in range(len(column_heights) - 1):
        bumpiness += abs(column_heights[i] - column_heights[i + 1])

    return bumpiness


# def calculate_holes(field: List[List[int]]) -> int:
#     """
#     For this function, a hole is defined if there is an empty space with a non-empty space directly above it.
#
#     0 1 0 0
#     0 0 0 0  -> # of holes = 1
#     0 0 0 0
#     """
#     column_holes = [0] * len(field[0])
#
#     for col in range(len(field[0])):
#         holes = 0
#
#         for row in range(1, len(field)):  # Start from the second row
#             if field[row][col] == 0 and field[row - 1][col] == 1:
#                 holes += 1
#
#         column_holes[col] = holes
#
#     return sum(column_holes)


def calculate_holes(field) -> int:
    """
    For this function, a hole is defined if there is an empty with at least a non-empty space above it

    0 1 0 0
    0 0 0 0  -> # of holes = 2
    0 0 0 0
    """
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


def game_over(field: List[List[int]]) -> bool:
    """
    Detect when it is Game Over. It is Game Over when the Game Over animation starts.
    """
    return all(cell == 1 for cell in field[0])


def get_best_field(playfield: List[List[int]], current_piece_val: int, num_cols=10):
    """
    Returns the best field out of all the possible fields based on playfield.
    The function chooses the best field by calculating the fitness of each playfield.
    Fitness is equal to (-0.510066 * aggregate_height) + (0.760666 * cleared_lines) + (-0.35663 * holes) + \
        (-0.184483 * bumpiness).

    >>> initial_field = [[0 for _ in range(10)] for _ in range(20)]
    >>> get_best_field(initial_field, 18, 10)
    (1, 18, 18)
    """
    possible_fields_list = get_possible_fields(playfield, current_piece_val, num_cols)
    max_fitness = -1000000
    field_index = 0

    for i in range(len(possible_fields_list)):

        possible_field_dict = possible_fields_list[i]

        aggregate_height = sum(possible_field_dict['col_heights'])
        # If there is all 1s in a row, cleared_lines += 1
        cleared_lines = 0
        for row in possible_field_dict['field']:
            if all(cell == 1 for cell in row):
                cleared_lines += 1
        holes = calculate_holes(possible_field_dict['field'])
        bumpiness = calculate_bumpiness(possible_field_dict['col_heights'])
        current_fitness = (-0.510066 * aggregate_height) + (0.760666 * cleared_lines) + (-0.35663 * holes) + \
                          (-0.184483 * bumpiness)

        if current_fitness > max_fitness:
            max_fitness = current_fitness
            field_index = i

    print('- ' * (len(possible_fields_list[field_index]['field'])))
    for row in possible_fields_list[field_index]['field']:
        print(' '.join(map(str, row)))
    print('- ' * (len(possible_fields_list[field_index]['field'])))

    return possible_fields_list[field_index]['best_col'], possible_fields_list[field_index]['best_piece'], \
        possible_fields_list[field_index]['best_rotation']


def move_piece(current_col, current_rotation, target_col, target_rotation) -> List[Dict[str, bool]]:
    """
    Returns a list of moves the AI should perform in order to move the current piece to the desired position and orientation.
    move_dict.keys() represent the available buttons on the NES controller.
    move_dict.values represent if the button should be pressed or not.

    Args:
        current_col (int): The current column of the piece.
        current_rotation (int): The current rotation of the piece.
        target_col (int): The target column for the piece.
        target_rotation (int): The target rotation for the piece.

    Returns:
        List[Dict[str, bool]]: A list of move dictionaries representing the sequence of button presses.

    >>> move_piece(18, 5, 18, 1, 18, 17)
    [{'B': False, 'null': False, 'SELECT': False, 'START': False, 'UP': False, 'DOWN': False, 'LEFT': True,
    'RIGHT': False, 'A': True}, {'B': False, 'null': False, 'SELECT': False, 'START': False, 'UP': False, 'DOWN': False,
     'LEFT': True, 'RIGHT': False, 'A': False}, {'B': False, 'null': False, 'SELECT': False, 'START': False,
     'UP': False, 'DOWN': False, 'LEFT': True, 'RIGHT': False, 'A': False}, {'B': False, 'null': False, 'SELECT': False,
      'START': False, 'UP': False, 'DOWN': False, 'LEFT': True, 'RIGHT': False, 'A': False}]
    """
    move_list = []

    # Rotate the piece if needed
    while current_rotation != target_rotation:
        move_dict = {"B": False, "null": False, "SELECT": False, "START": False, "UP": False, "DOWN": False,
                     "LEFT": False, "RIGHT": False, "A": False}
        if current_rotation < target_rotation:
            # move_dict["B"] = True
            move_dict["A"] = True
            current_rotation += 1
        elif current_rotation > target_rotation:
            # move_dict["A"] = True
            move_dict["B"] = True
            current_rotation -= 1
        move_list.append(move_dict)

    # Move the piece left or right
    while current_col != target_col:
        move_dict = {"B": False, "null": False, "SELECT": False, "START": False, "UP": False, "DOWN": False,
                     "LEFT": False, "RIGHT": False, "A": False}
        if current_col < target_col:
            move_dict["RIGHT"] = True
            current_col += 1
        elif current_col > target_col:
            move_dict["LEFT"] = True
            current_col -= 1
        move_list.append(move_dict)

    return move_list


env = retro.make('Tetris-Nes', state='StartLv5')
field_start_addr = 0x0400
num_rows = 20
num_cols = 10
imgarray = []


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        obs = env.reset()
        action = env.action_space.sample()
        inx, iny, inc = env.observation_space.shape
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        gameover = False
        prev_field = None
        frame = 0
        cleared_lines = 0

        while not gameover:
            frame += 1
            state = env.get_ram()
            current_piece = state[0x0062]
            next_piece = state[0x00BF]
            x_position = state[0x0040]
            y_position = state[0x0041]

            print('Current Piece:', current_piece)
            print('Next Piece:', next_piece)
            # print('Frame:', frame)

            # Uncomment to see NES Tetris running in emulator
            env.render()

            cropped_obs = obs[40:200, 88:168, :]  # Width = 168 - 88 = 80, Height = 200 - 40 = 160
            cropped_obs = cv2.resize(cropped_obs, (10, 20))
            cropped_obs = cv2.cvtColor(cropped_obs, cv2.COLOR_BGR2GRAY)
            # plt.imshow(cropped_obs)
            # plt.pause(1)
            # plt.clf()
            # imgarray = np.ndarray.flatten(cropped_obs)
            # nnOutput = net.activate(imgarray)

            field = read_field(env, field_start_addr, num_rows, num_cols)
            update_field = update_field_with_piece(field, current_piece, (x_position, y_position))

            # Prints playfield with piece
            # print("Playfield With Piece:")
            # print('* ' * (len(update_field[0])))
            # for row in update_field:
            #     print(' '.join(map(str, row)))
            # print('* ' * (len(update_field[0])))

            # best field or move piece needs to consider the current_pieces' position
            best_col, best_piece, best_rotation = get_best_field(field, current_piece, num_cols)
            # print(best_col, best_piece, best_rotation)
            print('X Position:', x_position)
            print('Best Column:', best_col)
            pivot_x, pivot_y = piece_pivot[current_piece]
            adjusted_x = x_position - pivot_y

            move_list = move_piece(adjusted_x + 1, current_piece, best_col, best_rotation)
            print('Move List:', move_list)

            if move_list == []:
                _, _, _, info = env.step([])  # Advance the environment without any inputs
            else:
                for move in move_list:
                    print('Current Move:', move)
                    if move:
                        _, _, _, info = env.step(list(move.values()))
                        # score = info['score']

            if game_over(update_field):
                gameover = True
            else:
                prev_field = [row[:] for row in update_field]

            if gameover:

                # If game over animation happens at last frame
                if all(cell == 1 for cell in field[0]):
                    column_heights = calculate_column_heights(prev_field)
                    aggregate_height = sum(column_heights)
                    print("Previous Column Heights:", column_heights)
                    print("Cleared Lines:", cleared_lines)
                    bumpiness = calculate_bumpiness(column_heights)
                    print("Previous Bumpiness:", bumpiness)
                    column_holes = calculate_holes(prev_field)
                    print("Previous Column Holes:", column_holes)
                    print("Gameover:", game_over(field))
                    print("Time Survived:", frame)
                    # print("Score:", score)

                else:
                    column_heights = calculate_column_heights(update_field)
                    aggregate_height = sum(column_heights)
                    print("Column Heights:", column_heights)
                    print("Cleared Lines:", cleared_lines)
                    bumpiness = calculate_bumpiness(column_heights)
                    print("Bumpiness:", bumpiness)
                    column_holes = calculate_holes(update_field)
                    print("Column Holes:", column_holes)
                    print("Gameover:", game_over(update_field))
                    print("Time Survived:", frame)
                    # print("Score:", score)

            genome.fitness = cleared_lines


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward-tetris')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
