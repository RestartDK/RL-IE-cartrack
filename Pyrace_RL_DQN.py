import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # to avoid some "memory" errors with TkAgg backend
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
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
VERSION_NAME = 'DQN_v01'  # the name for our model

REPORT_EPISODES = 500  # report (plot) every...
DISPLAY_EPISODES = 100  # display live game every...

# DQN Hyperparameters
MEMORY_SIZE = 10000  # size of replay buffer
GAMMA = 0.99  # discount factor
EPSILON_START = 1.0  # initial exploration rate
EPSILON_MIN = 0.01  # minimum exploration rate
EPSILON_DECAY = 0.995  # decay rate for exploration
BATCH_SIZE = 64  # batch size for training
LEARNING_RATE = 0.001  # learning rate for the optimizer
UPDATE_TARGET_EVERY = 5  # how often to update target network (in episodes)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
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
        self.action_counts = np.zeros(action_size)
        self.state_visits = {}  # Will store as string keys for state tuples
        self.episode_rewards = []
        self.episode_lengths = []

    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Add sample to replay memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, explore_rate=0.0):
        # Epsilon-greedy action selection
        if np.random.rand() <= explore_rate:
            action = random.randrange(self.action_size)
        else:
            state_tensor = np.array(state).reshape(1, self.state_size)
            act_values = self.model.predict(state_tensor, verbose=0)
            action = np.argmax(act_values[0])
        
        # Track action selection
        self.action_counts[action] += 1
        
        # Track state visitation (rounded to 2 decimals for discretization)
        state_key = tuple(np.round(state, 2))
        self.state_visits[state_key] = self.state_visits.get(state_key, 0) + 1
        
        return action

    def replay(self, batch_size):
        # Train the model with randomly sampled batch from memory
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Calculate Q values
        states_tensor = np.array(states)
        next_states_tensor = np.array(next_states)
        
        targets = self.model.predict(states_tensor, verbose=0)
        next_q_values = self.target_model.predict(next_states_tensor, verbose=0)
        
        # Track Q-value statistics before update
        self.q_values_stats.append({
            'mean': float(np.mean(next_q_values)),
            'max': float(np.max(next_q_values)),
            'min': float(np.min(next_q_values))
        })
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + GAMMA * np.amax(next_q_values[i])
        
        # Train the model and track loss
        history = self.model.fit(states_tensor, targets, epochs=1, verbose=0)
        self.loss_history.append(float(history.history['loss'][0]))
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
            means = [stat['mean'] for stat in self.q_values_stats]
            maxes = [stat['max'] for stat in self.q_values_stats]
            mins = [stat['min'] for stat in self.q_values_stats]
            
            print("\nQ-value Statistics:")
            print(f"Mean Q-value: {np.mean(means):.4f}")
            print(f"Max Q-value: {np.max(maxes):.4f}")
            print(f"Min Q-value: {np.min(mins):.4f}")
        
        # Action distribution
        total_actions = np.sum(self.action_counts)
        if total_actions > 0:
            print("\nAction Distribution:")
            for i in range(self.action_size):
                percentage = (self.action_counts[i] / total_actions) * 100
                print(f"Action {i}: {percentage:.1f}%")
        
        # State visitation analysis
        if self.state_visits:
            print("\nState Visitation:")
            most_visited = sorted(self.state_visits.items(), key=lambda x: x[1], reverse=True)[:5]
            print("Top 5 most visited states:")
            for state, count in most_visited:
                print(f"State {state}: visited {count} times")
        
        # Plot diagnostics if matplotlib is available
        try:
            plt.figure(figsize=(15, 5))
            
            # Loss plot
            plt.subplot(131)
            plt.plot(self.loss_history)
            plt.title('Training Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            
            # Q-value plot
            plt.subplot(132)
            plt.plot([stat['mean'] for stat in self.q_values_stats], label='Mean Q')
            plt.plot([stat['max'] for stat in self.q_values_stats], label='Max Q')
            plt.plot([stat['min'] for stat in self.q_values_stats], label='Min Q')
            plt.title('Q-value Evolution')
            plt.xlabel('Training Steps')
            plt.ylabel('Q-value')
            plt.legend()
            
            # Action distribution
            plt.subplot(133)
            plt.bar(range(self.action_size), self.action_counts)
            plt.title('Action Distribution')
            plt.xlabel('Action')
            plt.ylabel('Count')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not create plots: {e}")
            
        return {
            'loss_stats': {
                'mean': np.mean(self.loss_history) if self.loss_history else None,
                'min': np.min(self.loss_history) if self.loss_history else None,
                'max': np.max(self.loss_history) if self.loss_history else None
            },
            'q_value_stats': {
                'mean': np.mean(means) if self.q_values_stats else None,
                'min': np.min(mins) if self.q_values_stats else None,
                'max': np.max(maxes) if self.q_values_stats else None
            },
            'action_distribution': self.action_counts.tolist(),
            'most_visited_states': most_visited if self.state_visits else None
        }

def simulate(learning=True, episode_start=0):
    global agent
    
    env.set_view(True)
    explore_rate = agent.epsilon if learning else 0
    
    total_reward = 0
    total_rewards = []
    max_reward = -10_000
    
    for episode in range(episode_start, NUM_EPISODES + episode_start):
        if episode > 0:
            total_rewards.append(total_reward)
            
            if learning and episode % REPORT_EPISODES == 0:
                plt.plot(total_rewards)
                plt.ylabel('rewards')
                plt.show(block=False)
                plt.pause(4.0)
                
                # Save the model
                model_dir = f'models_{VERSION_NAME}'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                
                model_path = f'{model_dir}/dqn_model_{episode}.keras'
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
            
            # Display game
            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.set_msgs(['DQN SIMULATE',
                              f'Episode: {episode}',
                              f'Time steps: {t}',
                              f'check: {info["check"]}',
                              f'dist: {info["dist"]}',
                              f'crash: {info["crash"]}',
                              f'Reward: {total_reward:.0f}',
                              f'Max Reward: {max_reward:.0f}',
                              f'Epsilon: {agent.epsilon:.4f}'])
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
            print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon}")

def load_and_play(episode, learning=False):
    global agent
    
    model_path = f'models_{VERSION_NAME}/dqn_model_{episode}.keras'
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
    env = gym.make("Pyrace-v1").unwrapped  # skip the TimeLimit and OrderEnforcing default wrappers
    print('env', type(env))
    
    if not os.path.exists(f'models_{VERSION_NAME}'):
        os.makedirs(f'models_{VERSION_NAME}')
    
    # Get environment details
    STATE_SIZE = env.observation_space.shape[0]  # 5 radar readings
    ACTION_SIZE = env.action_space.n  # 3 actions
    print(f"State size: {STATE_SIZE}, Action size: {ACTION_SIZE}")
    
    # Initialize DQN agent
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    
    NUM_EPISODES = 35000  # Extended to allow even more learning
    MAX_T = 2000
    
    # Set to True to train from scratch, False to load a pre-trained model
    TRAIN_FROM_SCRATCH = False
    # Set to True to continue training a loaded model, False to just evaluate it
    CONTINUE_TRAINING = False  # We're now in evaluation mode
    # Latest episode to load - use your highest available model
    LATEST_EPISODE = 30000  # Using the latest trained model
    
    if TRAIN_FROM_SCRATCH:
        simulate(learning=True)
    else:
        load_and_play(episode=LATEST_EPISODE, learning=CONTINUE_TRAINING) 