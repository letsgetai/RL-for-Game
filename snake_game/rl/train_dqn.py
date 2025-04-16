from snake_env import SnakeEnv
from dqn import DQNAgent
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(rewards, save_path='training_curve.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(save_path)
    plt.close()

def main():
    # Initialize environment and agent
    env = SnakeEnv(grid_size=10)
    agent = DQNAgent(env)
    
    # Train the agent
    rewards = agent.train(num_episodes=1000)
    
    # Plot and save training results
    plot_training_results(rewards)
    
    print("Training completed. Model saved in 'models/dqn_best.pth'")

if __name__ == "__main__":
    main()
