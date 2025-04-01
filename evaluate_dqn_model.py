import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # to avoid some "memory" errors with TkAgg backend
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import gymnasium as gym
import gym_race

# Configuration
VERSION_NAME = 'DQN_v01'  # the name for our DQN model
VERSION_NAME_QT = 'QT_v02'  # the name for the Q-table model
episode_dqn = 5000  # Change this to match your trained DQN model episode
episode_qt = 3500   # Q-table model episode

# Paths
MODEL_PATH = f'models_{VERSION_NAME}/dqn_model_{episode_dqn}.keras'
QTABLE_PATH = f'models_{VERSION_NAME_QT}/q_table_{episode_qt}.npy'
MEMORY_PATH = f'models_{VERSION_NAME_QT}/memory_{episode_qt}.npy'

# Function to convert state to bucket (used for Q-table lookup)
def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        # Assuming state bounds are 0-10 and using 11 buckets
        bucket_index = min(10, max(0, int(state[i])))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def run_dqn_episode(env, model, render=True, max_steps=2000):
    """Run a single episode using the DQN model"""
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    crash = False
    max_distance = 0
    checkpoints = 0
    
    while not done and steps < max_steps:
        steps += 1
        
        # Convert observation to model input format
        state = np.array(obs).reshape(1, -1)
        
        # Predict action (no exploration)
        q_values = model.predict(state, verbose=0)
        action = np.argmax(q_values[0])
        
        # Take action
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        # Track metrics
        if info['dist'] > max_distance:
            max_distance = info['dist']
        
        checkpoints = info['check']
        crash = info['crash']
        
        if render:
            env.set_msgs(['DQN TEST',
                          f'Steps: {steps}',
                          f'Checkpoints: {checkpoints}',
                          f'Distance: {info["dist"]}',
                          f'Crash: {crash}',
                          f'Reward: {total_reward:.0f}'])
            env.render()
    
    return {
        'steps': steps,
        'reward': total_reward,
        'crash': crash,
        'max_distance': max_distance,
        'checkpoints': checkpoints
    }

def run_qtable_episode(env, q_table, render=True, max_steps=2000):
    """Run a single episode using the Q-table"""
    obs, _ = env.reset()
    state = state_to_bucket(obs)
    done = False
    total_reward = 0
    steps = 0
    crash = False
    max_distance = 0
    checkpoints = 0
    
    while not done and steps < max_steps:
        steps += 1
        
        # Select action based on Q-table (no exploration)
        action = np.argmax(q_table[state])
        
        # Take action
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        # Next state
        state = state_to_bucket(obs)
        
        # Track metrics
        if info['dist'] > max_distance:
            max_distance = info['dist']
        
        checkpoints = info['check']
        crash = info['crash']
        
        if render:
            env.set_msgs(['QTABLE TEST',
                          f'Steps: {steps}',
                          f'Checkpoints: {checkpoints}',
                          f'Distance: {info["dist"]}',
                          f'Crash: {crash}',
                          f'Reward: {total_reward:.0f}'])
            env.render()
    
    return {
        'steps': steps,
        'reward': total_reward,
        'crash': crash,
        'max_distance': max_distance,
        'checkpoints': checkpoints
    }

def compare_models(num_episodes=10):
    """Compare DQN and Q-table models over multiple episodes"""
    # Create environment
    env = gym.make("Pyrace-v1").unwrapped
    env.set_view(True)
    
    # Load DQN model
    try:
        dqn_model = load_model(MODEL_PATH)
        print(f"DQN model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading DQN model: {e}")
        return
    
    # Load Q-table
    try:
        q_table = np.load(QTABLE_PATH, allow_pickle=True)
        print(f"Q-table loaded from {QTABLE_PATH}")
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return
    
    # Run episodes for each model
    dqn_results = []
    qt_results = []
    
    for i in range(num_episodes):
        print(f"\nEpisode {i+1}/{num_episodes}")
        
        # DQN episode
        print("Running DQN model...")
        dqn_result = run_dqn_episode(env, dqn_model, render=True)
        dqn_result['model'] = 'DQN'
        dqn_results.append(dqn_result)
        print(f"DQN - Steps: {dqn_result['steps']}, Reward: {dqn_result['reward']:.1f}, Crash: {dqn_result['crash']}")
        
        # Q-table episode
        print("Running Q-table model...")
        qt_result = run_qtable_episode(env, q_table, render=True)
        qt_result['model'] = 'Q-table'
        qt_results.append(qt_result)
        print(f"Q-table - Steps: {qt_result['steps']}, Reward: {qt_result['reward']:.1f}, Crash: {qt_result['crash']}")
    
    # Combine results for analysis
    all_results = dqn_results + qt_results
    df = pd.DataFrame(all_results)
    
    # Calculate statistics by model
    stats = df.groupby('model').agg({
        'steps': ['mean', 'min', 'max'],
        'reward': ['mean', 'min', 'max'],
        'crash': lambda x: x.mean() * 100,  # Percentage of crashes
        'max_distance': ['mean', 'max'],
        'checkpoints': ['mean', 'max']
    })
    
    # Rename columns for readability
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.rename(columns={
        'crash_<lambda_0>': 'crash_percent'
    })
    
    print("\n===== Performance Comparison =====")
    print(stats)
    
    # Save results to CSV
    df.to_csv('model_comparison_results.csv', index=False)
    print("Detailed results saved to model_comparison_results.csv")
    
    # Create comparison plots
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    df.boxplot(column='reward', by='model')
    plt.title('Reward Comparison')
    plt.suptitle('')
    
    plt.subplot(1, 2, 2)
    df.boxplot(column='steps', by='model')
    plt.title('Steps Comparison')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("Comparison plot saved to model_comparison.png")

def analyze_state_responses():
    """Analyze how the models respond to different states"""
    # Load models
    try:
        dqn_model = load_model(MODEL_PATH)
        q_table = np.load(QTABLE_PATH, allow_pickle=True)
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Sample states to test
    sample_states = [
        [5, 5, 5, 5, 5],  # Balanced distances all around
        [1, 5, 5, 5, 5],  # Close to wall on left
        [5, 5, 1, 5, 5],  # Close to wall ahead
        [5, 5, 5, 5, 1],  # Close to wall on right
        [10, 10, 10, 10, 10],  # Far from all walls
        [1, 1, 1, 1, 1]   # Close to walls in all directions
    ]
    
    action_names = ['Accelerate', 'Turn Right', 'Turn Left']
    
    print("\n===== State Response Analysis =====")
    
    for i, state in enumerate(sample_states):
        print(f"\nState {i+1}: {state}")
        
        # DQN prediction
        dqn_q_vals = dqn_model.predict(np.array([state]), verbose=0)[0]
        dqn_action = np.argmax(dqn_q_vals)
        
        # Q-table lookup
        bucket = state_to_bucket(state)
        qtable_q_vals = q_table[bucket]
        qtable_action = np.argmax(qtable_q_vals)
        
        print(f"DQN Q-values: {dqn_q_vals}")
        print(f"DQN best action: {action_names[dqn_action]}")
        print(f"Q-table Q-values: {qtable_q_vals}")
        print(f"Q-table best action: {action_names[qtable_action]}")
        
        if dqn_action == qtable_action:
            print("MATCH: Both models chose the same action")
        else:
            print("MISMATCH: Models chose different actions")

def update_model_paths():
    """Update model and Q-table paths based on user input"""
    global episode_dqn, episode_qt, MODEL_PATH, QTABLE_PATH
    
    new_episode_dqn = int(input(f"Enter DQN episode number (current: {episode_dqn}): ") or episode_dqn)
    new_episode_qt = int(input(f"Enter Q-table episode number (current: {episode_qt}): ") or episode_qt)
    
    episode_dqn = new_episode_dqn
    episode_qt = new_episode_qt
    MODEL_PATH = f'models_{VERSION_NAME}/dqn_model_{episode_dqn}.keras'
    QTABLE_PATH = f'models_{VERSION_NAME_QT}/q_table_{episode_qt}.npy'
    print(f"Updated paths:\nDQN model: {MODEL_PATH}\nQ-table: {QTABLE_PATH}")
    return True

if __name__ == "__main__":
    while True:
        print("\n===== DQN Model Evaluation =====")
        print(f"DQN model path: {MODEL_PATH}")
        print(f"Q-table path: {QTABLE_PATH}")
        print("\nOptions:")
        print("1. Compare models over multiple episodes")
        print("2. Analyze state responses")
        print("3. Change model/q-table episode numbers")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1/2/3/4): ")
        
        if choice == '1':
            num_episodes = int(input("Number of episodes to compare (default 5): ") or 5)
            compare_models(num_episodes)
        elif choice == '2':
            analyze_state_responses()
        elif choice == '3':
            update_model_paths()
        elif choice == '4':
            print("Exiting program...")
            break
        else:
            print("Invalid choice") 