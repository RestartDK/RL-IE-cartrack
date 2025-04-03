import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # to avoid some "memory" errors with TkAgg backend
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import gymnasium as gym
import gym_race

def parse_arguments():
    parser = argparse.ArgumentParser(description='DQN Model Evaluation')
    parser.add_argument('--checkpoint', type=int, default=30000,
                       help='Checkpoint episode to load for evaluation')
    parser.add_argument('--display-every', type=int, default=1,
                       help='Display game every N episodes')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save evaluation plots to file')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save evaluation results to CSV')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering of the environment')
    return parser.parse_args()

# Configuration
VERSION_NAME = 'DQN_v01'  # the name for our DQN model
episode_dqn = 30000  # Change this to match your trained DQN model episode

# Paths
MODEL_PATH = f'models_{VERSION_NAME}/dqn_model_{episode_dqn}.keras'

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

def evaluate_model(num_episodes=10, render=True, save_plots=True, save_csv=True, display_every=1):
    """Evaluate DQN model over multiple episodes"""
    # Create environment
    env = gym.make("Pyrace-v1").unwrapped
    env.set_view(render)
    
    # Load DQN model
    try:
        dqn_model = load_model(MODEL_PATH)
        print(f"DQN model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading DQN model: {e}")
        return
    
    # Run episodes
    dqn_results = []
    
    for i in range(num_episodes):
        print(f"\nEpisode {i+1}/{num_episodes}")
        
        # DQN episode
        print("Running DQN model...")
        should_render = render and (i % display_every == 0)
        dqn_result = run_dqn_episode(env, dqn_model, render=should_render)
        dqn_result['episode'] = i + 1
        dqn_results.append(dqn_result)
        print(f"Episode {i+1} - Steps: {dqn_result['steps']}, Reward: {dqn_result['reward']:.1f}, Crash: {dqn_result['crash']}")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(dqn_results)
    
    # Calculate statistics
    stats = {
        'steps': {
            'mean': df['steps'].mean(),
            'min': df['steps'].min(),
            'max': df['steps'].max()
        },
        'reward': {
            'mean': df['reward'].mean(),
            'min': df['reward'].min(),
            'max': df['reward'].max()
        },
        'crash_rate': (df['crash'].mean() * 100),
        'max_distance': df['max_distance'].max(),
        'avg_checkpoints': df['checkpoints'].mean()
    }
    
    print("\n===== Performance Statistics =====")
    print(f"Average Steps: {stats['steps']['mean']:.1f} (min: {stats['steps']['min']}, max: {stats['steps']['max']})")
    print(f"Average Reward: {stats['reward']['mean']:.1f} (min: {stats['reward']['min']:.1f}, max: {stats['reward']['max']:.1f})")
    print(f"Crash Rate: {stats['crash_rate']:.1f}%")
    print(f"Max Distance: {stats['max_distance']:.1f}")
    print(f"Average Checkpoints: {stats['avg_checkpoints']:.1f}")
    
    # Save results
    if save_csv:
        df.to_csv('dqn_evaluation_results.csv', index=False)
        print("\nDetailed results saved to dqn_evaluation_results.csv")
    
    # Create plots
    if save_plots:
        plt.figure(figsize=(15, 5))
        
        # Reward plot
        plt.subplot(131)
        plt.plot(df['episode'], df['reward'])
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Steps plot
        plt.subplot(132)
        plt.plot(df['episode'], df['steps'])
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        # Distance plot
        plt.subplot(133)
        plt.plot(df['episode'], df['max_distance'])
        plt.title('Max Distance per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Distance')
        
        plt.tight_layout()
        plt.savefig('dqn_evaluation_plots.png')
        print("Plots saved to dqn_evaluation_plots.png")
    
    return df, stats

def analyze_state_responses(model=None):
    """Analyze how the model responds to different states"""
    if model is None:
        try:
            model = load_model(MODEL_PATH)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
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
    
    results = []
    for i, state in enumerate(sample_states):
        print(f"\nState {i+1}: {state}")
        
        # DQN prediction
        q_values = model.predict(np.array([state]), verbose=0)[0]
        action = np.argmax(q_values)
        
        results.append({
            'state': state,
            'q_values': q_values,
            'chosen_action': action_names[action]
        })
        
        print(f"Q-values: {q_values}")
        print(f"Chosen action: {action_names[action]}")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Update global model path based on checkpoint argument
    episode_dqn = args.checkpoint
    MODEL_PATH = f'models_{VERSION_NAME}/dqn_model_{episode_dqn}.keras'
    
    while True:
        print("\n===== DQN Model Evaluation =====")
        print(f"Model path: {MODEL_PATH}")
        print("\nOptions:")
        print("1. Evaluate model over multiple episodes")
        print("2. Analyze state responses")
        print("3. Change model episode number")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1/2/3/4): ")
        
        if choice == '1':
            num_episodes = int(input(f"Number of episodes to evaluate (default {args.episodes}): ") or args.episodes)
            df, stats = evaluate_model(
                num_episodes=num_episodes,
                render=not args.no_render,
                save_plots=args.save_plots,
                save_csv=args.save_csv,
                display_every=args.display_every
            )
        elif choice == '2':
            analyze_state_responses()
        elif choice == '3':
            new_episode = int(input(f"Enter episode number (current: {episode_dqn}): "))
            episode_dqn = new_episode
            MODEL_PATH = f'models_{VERSION_NAME}/dqn_model_{episode_dqn}.keras'
            print(f"Updated model path: {MODEL_PATH}")
        elif choice == '4':
            print("Exiting program...")
            break
        else:
            print("Invalid choice") 