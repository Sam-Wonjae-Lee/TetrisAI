# Tetris AI by Wonjae Lee

##**Project Goal**
The goal is to develop a program that can play classic Tetris by itself. The goal is achieved by using OpenAI's retro emulator that can run retro games within Python programs. 
The game can analyzed using the classic Tetris [Ram Map](https://datacrystal.romhacking.net/wiki/Tetris_(NES):RAM_map) which contains hexadecimal values of elements of the game such as lines cleared, current piece and the current board state.

##**Algorithm**
Here is how the program plays Tetris:
1. The program reads the current state of the board by extracting the RAM data. (The field start address is 0x0400)
2. Get all possible fields (board states) with the current piece based on the current board state and the current falling piece.
3. Calculate factors the program uses later like column heights, bumpiness, and holes of each possible field.
4. From the factors of each possible field, get the best field.
5. The program will move the current piece in the current board to match the best field.

##**TODOs**
- Algorithm is not smart enough to clear a tetris and do t spins
- Try different methods related to neural networks
