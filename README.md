# omf_projecti
My atempt to do RL project

## Project Structure

The project consists of the following main files:

- `critic.py`: Consolidated script for training and testing the reinforcement learning model.
- `Enviroment.py`: Script for capturing the game screen and preprocessing images.
- `model16.py`: Script defining the neural network architecture for the agent.
- `run.py`: Script for running the reinforcement learning agent.
- `requirements.txt`: List of required Python packages.

## How to Run

1. Get One Must Fall 2097.
2. Get DOSBox.
3. Run One Must Fall 2097 with DOSBox.
4. Go to the 1 player or 2 player menu.
5. Run the `run.py` script.

## Purpose

The purpose of this project is to train a reinforcement learning (RL) agent to play the game "One Must Fall 2097" using the DOSBox emulator. The project involves capturing the game screen, preprocessing the images, and using neural networks to predict actions and optimize the agent's performance.

## Detailed Instructions

1. **Install Dependencies**: Install the required Python packages by running `pip install -r requirements.txt`.
2. **Run the Game**: Start One Must Fall 2097 using DOSBox and navigate to the 1 player or 2 player menu.
3. **Run the Script**: Execute the `run.py` script to start the reinforcement learning agent.
4. **Training**: The agent will start training by capturing the game screen, preprocessing the images, and using the neural network to predict actions and optimize performance.
5. **Saving Models**: The trained models will be saved automatically using the `save_model` function.

## Functions and Classes

### `critic.py`

- **Critic Class**: Defines the neural network architecture for the critic model.
- **capture_screen Function**: Captures the game screen using PyAutoGUI and OpenCV.
- **tmp_matching Function**: Matches the current screen with a template image using OpenCV.
- **screen_preprocess Function**: Preprocesses the captured screen image.
- **optimize Function**: Updates the neural network weights based on the agent's performance.
- **save_model Function**: Saves the trained model.

### `Enviroment.py`

- **Enviroment Class**: Captures the game screen and preprocesses images.
- **capture_screen Function**: Captures the game screen using PyAutoGUI and OpenCV.
- **tmp_matching Function**: Matches the current screen with a template image using OpenCV.
- **screen_preprocess Function**: Preprocesses the captured screen image.
- **match Function**: Matches the current screen with predefined templates to determine the game state.
- **get_fight_status Function**: Retrieves the health status of the players during a fight.

### `model16.py`

- **Agent Class**: Defines the neural network architecture for the agent.
- **conv_block Function**: Defines a convolutional block for the neural network.
- **forward Function**: Defines the forward pass of the neural network.
- **init_hidden Function**: Initializes the hidden state for the LSTM layer.

### `run.py`

- **Replay_memory Class**: Implements replay memory to store and sample past experiences for training.
- **Fighter Class**: Defines the neural network architecture for the fighter model.
- **optimize Function**: Updates the neural network weights based on the agent's performance.
- **save_model Function**: Saves the trained model.
- **select_action Function**: Selects an action based on the current state and exploration-exploitation strategy.

## Notes

- The project uses PyTorch for building and training neural networks, OpenCV for image processing, and PyAutoGUI for interacting with the game.
- The `optimize` function has been refactored to improve modularity, readability, and robustness.
- The project captures the game screen using the `capture_screen` function and preprocesses the images using the `screen_preprocess` function.
- The project uses convolutional neural networks (CNNs) to process the game screen images and predict actions.
- The project uses reinforcement learning techniques to train the agent.
- The project uses replay memory to store and sample past experiences for training.
- The project saves the trained models using the `save_model` function.
