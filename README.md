# Flappy Bird AI

This project uses NEAT to train an AI to play Flappy Bird.

## Overview

- **Language:** Python  
- **Libraries:** pygame, neat-python, etc.
- **Description:** The AI is trained to play Flappy Bird by evolving a neural network.

## Setup

1. Clone this repository.
2. Create a virtual environment:

   ```zsh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the training:

   ```bash
   python main.py
   ```

## Files

- **main.py**: Contains the game logic and NEAT training.
- **replay_best.py**: Replays the best performing genome.
- **visualize.py**: Contains functions to visualize training statistics and network structure.
- **.gitignore**: Specifies files to ignore.

## Usage

- The training process runs for 50 generations.
- Visualization graphs and a pickle file with the best genome are generated.
