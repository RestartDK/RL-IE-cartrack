import sys, os
import math, random
import numpy as np
import matplotlib

from gym_race.envs.pyrace_2d import PyRace2D

matplotlib.use("Agg")  # to avoid some "memory" errors with TkAgg backend
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from collections import deque

import gymnasium as gym
import gym_race

"""
this imports race_env.py (a gym env) and pyrace_2d.py (the race game) and registers the env as "Pyrace-v1"

register(
    id='Pyrace-v1',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2_000,
)
"""
VERSION_NAME = "DQN_v02"  # the name for our model

# DQN Hyperparameters
MEMORY_SIZE = 10000  # Reduced memory size for faster training
GAMMA = 0.99  # discount factor
EPSILON_START = 1.0  # initial exploration rate
EPSILON_MIN = 0.1  # increased minimum exploration
EPSILON_DECAY = 0.95  # faster decay
BATCH_SIZE = 256  # larger batch size for Metal GPU
LEARNING_RATE = 0.001  # slightly higher learning rate
UPDATE_TARGET_EVERY = 3  # more frequent updates
MAX_T = 1000  # reduced max timesteps per episode

# Action space configuration
ACTION_LOW = -30.0  # minimum steering angle
ACTION_HIGH = 30.0  # maximum steering angle

# Training parameters (will be updated by command line arguments)
DISPLAY_EPISODES = 50  # display live game every...
REPORT_EPISODES = 100  # report (plot) every...
NUM_EPISODES = 1000  # reduced number of episodes


def parse_arguments():
    parser = argparse.ArgumentParser(description="DQN Training for Race Environment")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "both"],
        help="Mode to run the agent: train, eval, or both",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=3500,
        help="Checkpoint episode to load for evaluation",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to train"
    )
    parser.add_argument(
        "--display-every", type=int, default=100, help="Display game every N episodes"
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=500,
        help="Save model and plot every N episodes",
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disable rendering for faster training"
    )
    return parser.parse_args()


class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        
        # Initialize replay memory
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Main model - trained every step
        self.model = self._build_model()

        # Target model - used for predicting
        self.target_model = self._build_model()
        self.update_target_model()

        # Exploration rate
        self.epsilon = EPSILON_START

        # For statistics
        self.training_count = 0

        # Additional diagnostics
        self.loss_history = []
        self.q_values_stats = []
        self.action_distribution = []  # Store actual continuous actions
        self.state_visits = {}  # Will store as string keys for state tuples
        self.episode_rewards = []
        self.episode_lengths = []

    def _build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size),
            LeakyReLU(alpha=0.03),
            Dense(32),
            LeakyReLU(alpha=0.03),
            Dense(16),
            LeakyReLU(alpha=0.03),
            Dense(1, activation='tanh')  # Output single continuous value between -1 and 1
        ])
        model.compile(loss=Huber(), optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, explore_rate=0.0):
        if np.random.rand() <= explore_rate:
            # Random action between ACTION_LOW and ACTION_HIGH
            action = np.random.uniform(ACTION_LOW, ACTION_HIGH)
        else:
            state_tensor = np.array(state).reshape(1, self.state_size)
            # Model outputs value between -1 and 1, scale to actual range
            action = self.model.predict(state_tensor, verbose=0)[0][0]
            action = action * (ACTION_HIGH - ACTION_LOW) / 2 + (ACTION_HIGH + ACTION_LOW) / 2
        
        # Track action distribution
        self.action_distribution.append(action)

        # Track state visitation
        state_key = tuple(np.round(state, 2))
        self.state_visits[state_key] = self.state_visits.get(state_key, 0) + 1

        return np.array([action])  # Return as array for environment

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1][0] for experience in minibatch])  # Extract single value
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Normalize actions to [-1, 1] for network
        normalized_actions = (actions - (ACTION_HIGH + ACTION_LOW) / 2) / ((ACTION_HIGH - ACTION_LOW) / 2)

        # Calculate target Q values
        states_tensor = np.array(states)
        next_states_tensor = np.array(next_states)

        targets = normalized_actions.reshape(-1, 1)  # Current actions as initial targets
        next_q_values = self.target_model.predict(next_states_tensor, verbose=0)

        # Track Q-value statistics
        self.q_values_stats.append({
            "mean": float(np.mean(next_q_values)),
            "max": float(np.max(next_q_values)),
            "min": float(np.min(next_q_values))
        })

        for i in range(batch_size):
            if dones[i]:
                targets[i] = normalized_actions[i]
            else:
                targets[i] = normalized_actions[i] + GAMMA * (next_q_values[i] - normalized_actions[i])

        # Train the model and track loss
        history = self.model.fit(states_tensor, targets, epochs=1, verbose=0)
        self.loss_history.append(float(history.history["loss"][0]))
        self.training_count += 1

        # Decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def analyze_diagnostics(self):
        """Analyze the collected diagnostic information"""
        print("\n=== DQN Agent Diagnostics ===")

        # Loss analysis
        if self.loss_history:
            print("\nLoss Statistics:")
            print(f"Mean Loss: {np.mean(self.loss_history):.4f}")
            print(f"Min Loss: {np.min(self.loss_history):.4f}")
            print(f"Max Loss: {np.max(self.loss_history):.4f}")

        # Q-value analysis
        if self.q_values_stats:
            means = [stat["mean"] for stat in self.q_values_stats]
            maxes = [stat["max"] for stat in self.q_values_stats]
            mins = [stat["min"] for stat in self.q_values_stats]

            print("\nQ-value Statistics:")
            print(f"Mean Q-value: {np.mean(means):.4f}")
            print(f"Max Q-value: {np.max(maxes):.4f}")
            print(f"Min Q-value: {np.min(mins):.4f}")

        # Action distribution
        if self.action_distribution:
            print("\nAction Distribution:")
            for i, action in enumerate(self.action_distribution):
                print(f"Action {i}: {action:.2f}")

        # State visitation analysis
        if self.state_visits:
            print("\nState Visitation:")
            most_visited = sorted(
                self.state_visits.items(), key=lambda x: x[1], reverse=True
            )[:5]
            print("Top 5 most visited states:")
            for state, count in most_visited:
                print(f"State {state}: visited {count} times")

        # Plot diagnostics if matplotlib is available
        try:
            plt.figure(figsize=(15, 5))

            # Loss plot
            plt.subplot(131)
            plt.plot(self.loss_history)
            plt.title("Training Loss")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")

            # Q-value plot
            plt.subplot(132)
            plt.plot([stat["mean"] for stat in self.q_values_stats], label="Mean Q")
            plt.plot([stat["max"] for stat in self.q_values_stats], label="Max Q")
            plt.plot([stat["min"] for stat in self.q_values_stats], label="Min Q")
            plt.title("Q-value Evolution")
            plt.xlabel("Training Steps")
            plt.ylabel("Q-value")
            plt.legend()

            # Action distribution
            plt.subplot(133)
            plt.bar(range(len(self.action_distribution)), self.action_distribution)
            plt.title("Action Distribution")
            plt.xlabel("Action")
            plt.ylabel("Value")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Could not create plots: {e}")

        return {
            "loss_stats": {
                "mean": np.mean(self.loss_history) if self.loss_history else None,
                "min": np.min(self.loss_history) if self.loss_history else None,
                "max": np.max(self.loss_history) if self.loss_history else None,
            },
            "q_value_stats": {
                "mean": np.mean(means) if self.q_values_stats else None,
                "min": np.min(mins) if self.q_values_stats else None,
                "max": np.max(maxes) if self.q_values_stats else None,
            },
            "action_distribution": self.action_distribution,
            "most_visited_states": most_visited if self.state_visits else None,
        }


def simulate(learning=True, episode_start=0):
    global agent

    # Only set view if not using no-render flag
    env.set_view(not args.no_render)
    explore_rate = agent.epsilon if learning else 0

    total_reward = 0
    total_rewards = []
    max_reward = 10_000

    for episode in range(episode_start, NUM_EPISODES + episode_start):
        if episode > 0:
            total_rewards.append(total_reward)

            if learning and episode % REPORT_EPISODES == 0:
                plt.plot(total_rewards)
                plt.ylabel("rewards")
                plt.show(block=False)
                plt.pause(4.0)

                # Save the model
                model_dir = f"models_{VERSION_NAME}"
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                model_path = f"{model_dir}/dqn_model_{episode}.keras"
                agent.model.save(model_path)
                print(f"Model saved to {model_path}")

                plt.close()  # to avoid memory errors...

        # Reset environment
        obv, _ = env.reset()
        state = np.array(obv)
        total_reward = 0

        if not learning:
            env.pyrace.mode = 2  # continuous display of game

        for t in range(MAX_T):
            # Select action
            action = agent.act(state, explore_rate if learning else 0)

            # Take action
            next_obv, reward, done, _, info = env.step(action)
            next_state = np.array(next_obv)

            # Remember experience
            if learning:
                agent.remember(state, action, reward, next_state, done)

            total_reward += reward

            # Display game only if not using no-render flag
            if not args.no_render and (
                (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2)
            ):
                env.set_msgs(
                    [
                        "DQN SIMULATE",
                        f"Episode: {episode}",
                        f"Time steps: {t}",
                        f'check: {info["check"]}',
                        f'dist: {info["dist"]}',
                        f'crash: {info["crash"]}',
                        f"Reward: {total_reward:.0f}",
                        f"Max Reward: {max_reward:.0f}",
                        f"Epsilon: {agent.epsilon:.4f}",
                    ]
                )
                env.render()

            # Move to next state
            state = next_state

            # Train the agent by replaying experiences
            if learning and len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

            if done or t >= MAX_T - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                break

        # Update target model periodically
        if learning and episode % UPDATE_TARGET_EVERY == 0:
            agent.update_target_model()
            print(f"Target model updated at episode {episode}")

        # Print episode summary
        if episode % 10 == 0:
            print(
                f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon}"
            )


def load_and_play(episode, learning=False):
    global agent

    model_path = f"models_{VERSION_NAME}/dqn_model_{episode}.keras"
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        agent.model = loaded_model
        agent.update_target_model()
        print(f"Model loaded from {model_path}")

        # Run diagnostics before simulation
        print("\nInitial Model Diagnostics:")
        agent.analyze_diagnostics()

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Play game
    simulate(learning, episode)

    # Run diagnostics after simulation
    print("\nPost-Simulation Diagnostics:")
    agent.analyze_diagnostics()


if __name__ == "__main__":
    args = parse_arguments()
    DISPLAY_EPISODES = args.display_every
    REPORT_EPISODES = args.report_every
    NUM_EPISODES = args.episodes

    # Initialize environment
    env = gym.make("Pyrace-v1").unwrapped
    print("env", type(env))
    env.set_view(not args.no_render)

    if not os.path.exists(f"models_{VERSION_NAME}"):
        os.makedirs(f"models_{VERSION_NAME}")

    # Get environment details
    STATE_SIZE = env.observation_space.shape[0]  # 5 radar readings
    print(f"State size: {STATE_SIZE}")

    # Initialize DQN agent
    agent = DQNAgent(STATE_SIZE)

    def train_episode(episode, learning=True):
        total_reward = 0
        state = env.reset()[0]  # Get initial state
        done = False
        explore_rate = agent.epsilon if learning else 0

        for t in range(MAX_T):
            # Select action
            action = agent.act(state, explore_rate if learning else 0)

            # Take action
            next_state, reward, done, _, _ = env.step(action)

            # Remember experience
            if learning:
                agent.remember(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if learning and len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

            if done:
                break

        # Update target network periodically
        if learning and episode % UPDATE_TARGET_EVERY == 0:
            agent.update_target_model()

        return total_reward, t + 1

    def evaluate_agent(agent, num_episodes=5):
        print("\nEvaluating agent...")
        eval_rewards = []
        for i in range(num_episodes):
            total_reward, steps = train_episode(i, learning=False)
            eval_rewards.append(total_reward)
            print(f"Evaluation episode {i + 1}: Reward = {total_reward}, Steps = {steps}")
        return np.mean(eval_rewards)

    # Training loop
    if args.mode in ["train", "both"]:
        print("Training agent...")
        episode_rewards = []
        best_reward = float("-inf")

        for episode in range(NUM_EPISODES):
            total_reward, steps = train_episode(episode)
            episode_rewards.append(total_reward)

            # Store episode statistics
            agent.episode_rewards.append(total_reward)
            agent.episode_lengths.append(steps)

            if episode % DISPLAY_EPISODES == 0:
                print(
                    f"Episode {episode}: Reward = {total_reward}, Steps = {steps}, Epsilon = {agent.epsilon:.2f}"
                )

            if episode % REPORT_EPISODES == 0:
                # Save model
                agent.model.save(f"models_{VERSION_NAME}/dqn_model_{episode}.keras")

                # Plot training progress
                plt.figure(figsize=(10, 5))
                plt.plot(episode_rewards)
                plt.title("DQN Training Progress")
                plt.xlabel("Episode")
                plt.ylabel("Total Reward")
                plt.savefig(f"models_{VERSION_NAME}/training_progress_{episode}.png")
                plt.close()

                # Save if best model
                if total_reward > best_reward:
                    best_reward = total_reward
                    agent.model.save(f"models_{VERSION_NAME}/dqn_model_best.keras")

    # Evaluation
    if args.mode in ["eval", "both"]:
        print("\nLoading best model for evaluation...")
        checkpoint_path = f"models_{VERSION_NAME}/dqn_model_{args.checkpoint}.keras"
        if os.path.exists(checkpoint_path):
            agent.model.load_weights(checkpoint_path)
            mean_reward = evaluate_agent(agent)
            print(f"\nMean evaluation reward: {mean_reward}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

    # Final analysis
    agent.analyze_diagnostics()
