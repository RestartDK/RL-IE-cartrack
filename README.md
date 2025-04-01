# Deep Q-Network (DQN) for Race Car Environment

This project implements a Deep Q-Network (DQN) reinforcement learning algorithm to train an agent to navigate a 2D racing environment. The project converts an original Q-table implementation to use a neural network approach.

## Project Overview

The agent learns to drive a car around a racing track without crashing into walls. The implementation uses:

1. **Deep Q-Network (DQN)** - A neural network that approximates the Q-function
2. **Experience Replay** - Stores and randomly samples past experiences for training
3. **Target Network** - A separate network for stable Q-value targets

## Repository Structure

- `Pyrace_RL_DQN.py` - Main implementation of the DQN algorithm
- `Pyrace_RL_QTable.py` - Original Q-table implementation (for comparison)
- `evaluate_dqn_model.py` - Script to evaluate and compare DQN vs Q-table performance
- `gym_race/` - The racing environment
- `models_DQN_v01/` - Saved DQN model checkpoints
- `models_QT_v02/` - Saved Q-table checkpoints
- `requirements.txt` - Required Python packages

## Key Features

- **Neural Network Architecture**:
  - Input: 5 radar distance readings (state)
  - Hidden layers: 2 layers with 24 neurons each
  - Output: Q-values for 3 possible actions (accelerate, turn left, turn right)

- **Training Process**:
  - Trained for 30,000+ episodes
  - Used epsilon-greedy exploration strategy
  - Periodically updated target network

- **Performance**:
  - Successfully learned to navigate the entire track
  - Achieves maximum reward of 10,000 (complete lap)

## Results

The DQN agent successfully learned to navigate the racing track, improving from initial random movements to completing full laps without crashing. The model demonstrates:

1. Effective obstacle avoidance
2. Smooth cornering
3. Consistent lap completion

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. To run the DQN model in evaluation mode:
   ```
   python Pyrace_RL_DQN.py
   ```

3. To compare DQN with Q-table:
   ```
   python evaluate_dqn_model.py
   ```

## Training Your Own Model

To train the model from scratch, set `TRAIN_FROM_SCRATCH = True` in `Pyrace_RL_DQN.py`.

To continue training from a checkpoint, set:
```python
TRAIN_FROM_SCRATCH = False
CONTINUE_TRAINING = True
LATEST_EPISODE = [episode_number]  # e.g., 5000, 10000, etc.
```

## Acknowledgments

- This project was created as part of a reinforcement learning assignment
- Built using Gymnasium (formerly OpenAI Gym) for the environment
- Implemented with TensorFlow/Keras for the neural network 