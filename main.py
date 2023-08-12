import retro
import neat
import pickle
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict
import copy
import matplotlib.pyplot as plt

import pieces


# field_end_addr = 0x04C7
# current_piece_addr = 0x0062
# next_piece_addr = 0x00BF
# "buttons": ["B", null, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]


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


def calculate_holes(field: List[List[int]]) -> int:
    """
    For this function, a hole is defined if there is an empty space with a non-empty space directly above it.

    0 1 0 0
    0 0 0 0  -> # of holes = 1
    0 0 0 0
    """
    column_holes = [0] * len(field[0])

    for col in range(len(field[0])):
        holes = 0

        for row in range(1, len(field)):  # Start from the second row
            if field[row][col] == 0 and field[row - 1][col] == 1:
                holes += 1

        column_holes[col] = holes

    return sum(column_holes)


# def calculate_holes(field) -> int:
#     """
#     For this function, a hole is defined if there is an empty with at least a non-empty space above it
#
#     0 1 0 0
#     0 0 0 0  -> # of holes = 2
#     0 0 0 0
#     """
#     # Initialize the list to store the number of holes in each column
#     column_holes = [0] * len(field[0])
#
#     # Iterate through each column
#     for col in range(len(field[0])):
#         holes = 0
#         block_above = False
#
#         for row in range(len(field)):
#             if field[row][col] == 1:  # Check for an occupied cell
#                 block_above = True
#             elif field[row][col] == 0 and block_above:
#                 # Count the number of empty cells (holes) below the current cell in the column
#                 holes += 1
#         column_holes[col] = holes
#     return sum(column_holes)


def game_over(field: List[List[int]]) -> bool:
    """
    Detect when it is Game Over. It is Game Over when the Game Over animation starts.
    """
    return any(cell == 1 for cell in field[0])


def get_best_field(playfield: List[List[int]], current_piece_val: int, num_cols=10):
    """
    Returns the best field out of all the possible fields based on playfield.
    The function chooses the best field by calculating the fitness of each playfield.
    Fitness is equal to (-0.510066 * aggregate_height) + (0.760666 * cleared_lines) + (-0.35663 * holes) + \
        (-0.184483 * bumpiness).
    """
    possible_fields = pieces.get_possible_fields(playfield, current_piece_val, num_cols)
    max_fitness = -1000000
    field_index = 0

    for i in range(len(possible_fields)):

        # field = possible_fields[i]
        # print('* ' * (len(field[0])))
        # for row in field:
        #     print(' '.join(map(str, row)))
        # print('* ' * (len(field[0])))

        aggregate_height = sum(calculate_column_heights(possible_fields[i]))
        # If there is all 1s in a row, cleared_lines += 1
        cleared_lines = 0
        for row in possible_fields[i]:
            if all(cell == 1 for cell in row):
                cleared_lines += 1
        holes = calculate_holes(possible_fields[i])
        bumpiness = calculate_bumpiness(calculate_column_heights(possible_fields[i]))
        current_fitness = (-0.510066 * aggregate_height) + (0.760666 * cleared_lines) + (-0.35663 * holes) + \
                          (-0.184483 * bumpiness)
        if current_fitness > max_fitness:
            max_fitness = current_fitness
            field_index = i

    return possible_fields[field_index]


def move_piece(best_col, piece, rotation) -> List[Dict[str, bool]]:
    """
    Returns a list of moves the AI should perform in order.
    move_dict.keys() represents the availible buttons on the NES controller.
    move_dict.values represents if the button were pressed or not.
    best_col, piece and rotation are from getting the best move from potential fields.

    >>> move_piece(5, 2, 1)
    [{'B': True, 'null': False, 'SELECT': False, 'START': False, 'UP': False, 'DOWN': False, 'LEFT': True, 'RIGHT':
    False, 'A': False}]
    >>> move_piece(5, 2, 0)
    [{'B': True, 'null': False, 'SELECT': False, 'START': False, 'UP': False, 'DOWN': False, 'LEFT': False, 'RIGHT':
    False, 'A': False}, {'B': True, 'null': False, 'SELECT': False, 'START': False, 'UP': False, 'DOWN': False, 'LEFT':
    False, 'RIGHT': False, 'A': False}]
    """
    start_col = pieces.piece_column[rotation]
    move_list = []
    while best_col != start_col or piece != rotation:
        move_dict = {"B": False, "null": False, "SELECT": False, "START": False, "UP": False, "DOWN": False,
                     "LEFT": False, "RIGHT": False, "A": False}
        # For rotation
        if piece > rotation:
            move_dict["B"] = True
            rotation += 1
        elif piece < rotation:
            move_dict["A"] = True
            rotation -= 1

        # For moving left or right
        if best_col < start_col:
            move_dict["LEFT"] = True
            best_col += 1
        elif best_col > start_col:
            move_dict["RIGHT"] = True
            best_col -= 1

        move_list.append(move_dict)

    return move_list


# for i in range(1000):
#     move_list = []
#     action = move_list[i]
#     for values in action.values():
#         env.step(values)

# env = retro.make('Tetris-Nes', state='StartLv0')
# obs = env.reset()
# # move_list = env.action_space
# # print(move_list)
#
# for i in range(1000):
#     env.render()
#     # action = env.action_space.sample()
#     # print(action)
#     action = [0, 0, 0, 0, 0, 0, 0, 0, 1]
#     env.step(action)
#     # move_list = [1, 0, 1, 0, 0, 0, 1, 0, 1]
#     # observation, reward, done, info = env.step(move_list)
# env.close()

# ===================================== For Testing ========================================

# def main():
#     env = retro.make('Tetris-Nes', state='StartLv0')
#     field_start_addr = 0x0400
#     num_rows = 20
#     num_cols = 10
#
#     obs = env.reset()
#     done = False
#     prev_field = None
#
#     cropped_obs = obs[40:200, 88:168, :]
#
#     while not done:
#
#         # plt.imshow(cropped_obs)
#         # plt.pause(1)
#         # plt.clf()
#         field = read_field(env, field_start_addr, num_rows, num_cols)
#         # This does not work since both variables refer to same object
#         # prev_field = field
#         print('* ' * (len(field[0])))
#         for row in field:
#             print(' '.join(map(str, row)))
#         print('* ' * (len(field[0])))
#
#         # Check for game over before taking any action
#         if game_over(field) and prev_field is None:
#             print("Game Over!")
#             # Save the previous field state before the game over animation starts
#             prev_field = copy.deepcopy(field)
#             done = True
#
#         if done:
#             break
#
#         column_heights = calculate_column_heights(field)
#         print("Column Heights:", column_heights)
#         bumpiness = calculate_bumpiness(column_heights)
#         print("Bumpiness:", bumpiness)
#         column_holes = calculate_holes(field)
#         print("Column Holes:", column_holes)
#
#         action = env.action_space.sample()
#         obs, _, done, _ = env.step(action)
#         env.render()
#
#     env.close()
#
#     if prev_field is not None:
#         # Do something with the previous field state before game over
#         # For example, you can save it to a file, analyze, or use it for training.
#         print("Previous field state before game over:")
#         for row in prev_field:
#             print(' '.join(map(str, row)))
#         column_heights = calculate_column_heights(prev_field)
#         print("Previous Column Heights:", column_heights)
#         bumpiness = calculate_bumpiness(column_heights)
#         print("Previous Bumpiness:", bumpiness)
#         column_holes = calculate_holes(prev_field)
#         print("Previous Column Holes:", column_holes)
#
#
# if __name__ == '__main__':
#     main()


# ===================================== Run Single Instance ========================================

# env = retro.make('Tetris-Nes', state='StartLv0')
# field_start_addr = 0x0400
# num_rows = 20
# num_cols = 10
#
# imgarray = []
#
#
# def eval_genomes(genomes, config):
#     for genome_id, genome in genomes:
#         obs = env.reset()
#         action = env.action_space.sample()
#         inx, iny, inc = env.observation_space.shape
#         # print(inx, iny)
#         # inx = int(inx / 8)
#         # iny = int(iny / 8)
#         net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
#         current_max_fitness = 0
#         fitness_current = 0
#         frame = 0
#         gameover = False
#         prev_field = None
#
#         while not gameover:
#             state = env.get_ram()
#             current_piece = state[0x0062]
#             next_piece = state[0x00BF]
#
#             print('Current Piece:', current_piece)
#             print('Next Piece:', next_piece)
#
#             # Uncomment to see NES Tetris running in emulator
#             env.render()
#             frame += 1
#
#             # buttons = env.buttons
#             # active_buttons = [env.buttons[i] for i, button in enumerate(buttons) if button]
#             # print("All button states:", buttons)
#             # print("Active buttons:", active_buttons)
#
#             cropped_obs = obs[40:200, 88:168, :]    # Width = 168 - 88 = 80, Height = 200 - 40 = 160
#             # cropped_obs = cv2.resize(cropped_obs, (inx, iny))
#             cropped_obs = cv2.resize(cropped_obs, (10, 20))
#             # print(cropped_obs.shape)    # (30, 28, 3)
#             cropped_obs = cv2.cvtColor(cropped_obs, cv2.COLOR_BGR2GRAY)
#             # print(cropped_obs.shape)    # (30, 28)
#             # cropped_obs = np.reshape(cropped_obs, (inx, iny))
#             # print(cropped_obs.shape)    # (28, 30)
#             # plt.imshow(cropped_obs)
#             # plt.pause(1)
#             # plt.clf()
#             imgarray = np.ndarray.flatten(cropped_obs)
#
#             # obs = cv2.resize(obs, (inx, iny))
#             # obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
#             # obs = np.reshape(obs, (inx, iny))
#             # imgarray = np.ndarray.flatten(obs)
#
#             nnOutput = net.activate(imgarray)
#
#             obs, rew, done, info = env.step(nnOutput)
#
#             cleared_lines = info['cleared_lines']
#             score = info['score']
#
#             # current_piece = info['current_piece']
#             # next_piece = info['next_piece']
#
#             # print('Current Piece:', current_piece)
#             # print('Next Piece:', next_piece)
#
#             field = read_field(env, field_start_addr, num_rows, num_cols)
#             print(field)
#
#             if game_over(field):
#                 gameover = True
#             else:
#                 prev_field = [row[:] for row in field]
#
#             if gameover:
#                 # Print final field
#                 print('* ' * (len(field[0])))
#                 for row in field:
#                     print(' '.join(map(str, row)))
#                 print('* ' * (len(field[0])))
#
#                 # Print previous field one frame before game over
#                 print("Previous Field (when game over):")
#                 print('* ' * (len(prev_field[0])))
#                 for row in prev_field:
#                     print(' '.join(map(str, row)))
#                 print('* ' * (len(prev_field[0])))
#
#                 # If game over animation happens at last frame
#                 if all(cell == 1 for cell in field[0]):
#                     column_heights = calculate_column_heights(prev_field)
#                     aggregate_height = sum(column_heights)
#                     print("Previous Column Heights:", column_heights)
#                     print("Cleared Lines:", cleared_lines)
#                     bumpiness = calculate_bumpiness(column_heights)
#                     print("Previous Bumpiness:", bumpiness)
#                     column_holes = calculate_holes(prev_field)
#                     print("Previous Column Holes:", column_holes)
#                     print("Gameover:", game_over(field))
#                     print("Time Survived:", frame)
#                     print("Score:", score)
#
#                 else:
#                     column_heights = calculate_column_heights(field)
#                     aggregate_height = sum(column_heights)
#                     print("Column Heights:", column_heights)
#                     print("Cleared Lines:", cleared_lines)
#                     bumpiness = calculate_bumpiness(column_heights)
#                     print("Bumpiness:", bumpiness)
#                     column_holes = calculate_holes(field)
#                     print("Column Holes:", column_holes)
#                     print("Gameover:", game_over(field))
#                     print("Time Survived:", frame)
#                     print("Score:", score)
#
#                 fitness_current = (-0.860 * aggregate_height) + (0.433 * cleared_lines) + \
#                                   (-0.824 * column_holes) + (-0.343 * bumpiness) + (0.01 * frame) + \
#                                   (0.01 * score)
#
#                 if fitness_current > current_max_fitness:
#                     current_max_fitness = fitness_current
#                 print(genome_id, fitness_current)
#
#             genome.fitness = fitness_current
#
#
# config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                      'config-feedforward-tetris')
#
# p = neat.Population(config)
#
# p.add_reporter(neat.StdOutReporter(True))
# stats = neat.StatisticsReporter()
# p.add_reporter(stats)
# p.add_reporter(neat.Checkpointer(10))
#
# winner = p.run(eval_genomes)
#
# with open('winner.pkl', 'wb') as output:
#     pickle.dump(winner, output, 1)


# ===================================== Parallelization: Run Multiple Instances ========================================

# class Worker(object):
#     def __init__(self, genome, config):
#         self.genome = genome
#         self.config = config
#
#     def work(self):
#         self.env = retro.make('Tetris-Nes', state='StartLv0')
#         obs = self.env.reset()
#         action = self.env.action_space.sample()
#         inx, iny, inc = self.env.observation_space.shape
#         # inx = int(inx / 8)
#         # iny = int(iny / 8)
#
#         net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)
#         fitness = 0
#         frame = 0
#         gameover = False
#         prev_field = None
#
#         field_start_addr = 0x0400
#         num_rows = 20
#         num_cols = 10
#
#         while not gameover:
#             # Uncomment to see NES Tetris running in emulator
#             # self.env.render()
#             frame += 1
#             # obs = cv2.resize(obs, (inx, iny))
#             # obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
#             # obs = np.reshape(obs, (inx, iny))
#             # imgarray = np.ndarray.flatten(obs)
#             cropped_obs = obs[40:200, 88:168, :]
#             cropped_obs = cv2.resize(cropped_obs, (10, 20))
#             cropped_obs = cv2.cvtColor(cropped_obs, cv2.COLOR_BGR2GRAY)
#             imgarray = np.ndarray.flatten(cropped_obs)
#             # plt.imshow(cropped_obs)
#             # plt.pause(1)
#             # plt.clf()
#
#             nnOutput = net.activate(imgarray)
#             obs, rew, done, info = self.env.step(nnOutput)
#
#             cleared_lines = info['cleared_lines']
#             score = info['score']
#             field = read_field(self.env, field_start_addr, num_rows, num_cols)
#
#             if game_over(field):
#                 gameover = True
#             else:
#                 prev_field = [row[:] for row in field]
#
#             if gameover:
#
#                 # For previous frame
#                 if all(cell == 1 for cell in field[0]):
#                     # Print previous field one frame before game over
#                     print("Previous Field:")
#                     print('* ' * (len(prev_field[0])))
#                     for row in prev_field:
#                         print(' '.join(map(str, row)))
#                     print('* ' * (len(prev_field[0])))
#
#                     column_heights = calculate_column_heights(prev_field)
#                     aggregate_height = sum(column_heights)
#                     print("Previous Column Heights:", column_heights)
#                     print("Cleared Lines:", cleared_lines)
#                     bumpiness = calculate_bumpiness(column_heights)
#                     print("Previous Bumpiness:", bumpiness)
#                     column_holes = calculate_holes(prev_field)
#                     print("Previous Column Holes:", column_holes)
#
#                     print("Time Survived:", frame)
#                     print("Score:", score)
#                     print("Gameover:", game_over(field))
#                 # For final frame
#                 else:
#                     # Print final field
#                     print("Final Field:")
#                     print('* ' * (len(field[0])))
#                     for row in field:
#                         print(' '.join(map(str, row)))
#                     print('* ' * (len(field[0])))
#
#                     column_heights = calculate_column_heights(field)
#                     aggregate_height = sum(column_heights)
#                     print("Column Heights:", column_heights)
#                     print("Cleared Lines:", cleared_lines)
#                     bumpiness = calculate_bumpiness(column_heights)
#                     print("Bumpiness:", bumpiness)
#                     column_holes = calculate_holes(field)
#                     print("Column Holes:", column_holes)
#
#                     print("Time Survived:", frame)
#                     print("Score:", score)
#                     print("Gameover:", game_over(field))
#
#                 fitness = (-0.1 * aggregate_height) + (15 * cleared_lines) + \
#                           (-0.25 * column_holes) + (-0.25 * bumpiness) + (5 * score)
#
#         print(fitness)
#         return fitness
#
#
# def eval_genomes(genome, config):
#     worky = Worker(genome, config)
#     return worky.work()
#
#
# config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                      'config-feedforward-tetris')
# p = neat.Population(config)
# p.add_reporter(neat.StdOutReporter(True))
# stats = neat.StatisticsReporter()
# p.add_reporter(stats)
# p.add_reporter(neat.Checkpointer(10))
#
# if __name__ == '__main__':
#     pe = neat.ParallelEvaluator(10, eval_genomes)
#     winner = p.run(pe.evaluate)
#     with open('winner.pkl', 'wb') as output:
#         pickle.dump(winner, output, 1)
