# Tetris AI by Wonjae Lee
AI that playes classic Tetris.

## **Project Goal**
The goal is to develop a program that can play classic Tetris by itself. The goal is achieved by using OpenAI's retro emulator that can run retro games within Python programs. 
The game can analyzed using the classic Tetris [Ram Map](https://datacrystal.romhacking.net/wiki/Tetris_(NES):RAM_map) which contains hexadecimal values of elements of the game such as lines cleared, current piece and the current board state.

### Introduction, Libraries & Files
This program uses neat-python and gym-retro libraries to have the algorithm learn to play classic Tetris.
The program did not know anything about the controls but eventually came to understand the game.

* [OpenAI gym-retro Library](https://openai.com/research/gym-retro):\
  Emulator on Python that can run over 1000 games from several systems. It is a tool for reinforcement learning research on these games.
* Tetris-Nes Rom File:\
  A puzzle video game released for the NES. A rom file of the game is required and should be [imported](https://retro.readthedocs.io/en/latest/getting_started.html#importing-roms). 
* [NEAT-Python Library](https://neat-python.readthedocs.io/en/latest/):\
  NEAT (Neuroevolution of Augmening Topologies) is a genetic algorithm for evolving artificial neural networks. The next generation is produced by reproduction and mutation based on the fitness score (how well the network performs) of the previous generation. Neat-Python implements NEAT into Python in a convinient method.
* config-feedforward-tetris File:\
  Configuration file for the neural network.  \
  NOTE: When continuing training from a checkpoint file, if the config file is changed, it will not train properly. In other words, if you want to train the program with different configuration, you would have to train the program from the first generation.
* data.json File:\
...
* main.py File:\
...

## **Algorithm**
Here is how the program plays Tetris:
1. The program reads the current state of the board by extracting the RAM data. (The field start address is 0x0400)
2. Get all possible fields (board states) with the current piece based on the current board state and the current falling piece.
3. Calculate factors the program uses later like column heights, bumpiness, and holes of each possible field.
4. From the factors of each possible field, get the best field.
5. The program will move the current piece in the current board to match the best field.


