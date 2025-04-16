import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import json
from datetime import datetime

class SimpleDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(SimpleDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, 1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = x.float() / 3.0
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def append(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))

class DQNAgent:
    def __init__(self, env, learning_rate=1e-3, gamma=0.99, buffer_size=10000,
                 batch_size=64, target_update=100, epsilon_decay=0.97):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.net = SimpleDQN((env.grid_size, env.grid_size), env.action_space.n).to(self.device)
        self.target_net = SimpleDQN((env.grid_size, env.grid_size), env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        
        self.buffer = ExperienceBuffer(buffer_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_v = torch.FloatTensor(state).to(self.device)
            state_v = state_v.unsqueeze(0).unsqueeze(0)
            q_vals = self.net(state_v)
            return int(torch.argmax(q_vals))
    
    def train(self, num_episodes=1000, early_stop_reward=15.0):
        rewards_history = []
        best_reward = float('-inf')
        patience = 50
        episodes_without_improvement = 0
        training_start_time = datetime.now()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.append((state, action, reward, next_state, done))
                total_reward += reward
                state = next_state
                
                if len(self.buffer.buffer) >= self.batch_size:
                    states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                    
                    states_v = torch.FloatTensor(states).to(self.device).unsqueeze(1)
                    next_states_v = torch.FloatTensor(next_states).to(self.device).unsqueeze(1)
                    actions_v = torch.LongTensor(actions).to(self.device)
                    rewards_v = torch.FloatTensor(rewards).to(self.device)
                    done_mask = torch.BoolTensor(dones).to(self.device)
                    
                    current_q = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
                    next_q = self.target_net(next_states_v).max(1)[0]
                    next_q[done_mask] = 0.0
                    expected_q = rewards_v + self.gamma * next_q
                    
                    loss = nn.MSELoss()(current_q, expected_q.detach())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.net.state_dict())
            
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
            
            if total_reward > best_reward:
                best_reward = total_reward
                episodes_without_improvement = 0
                self.save('models/dqn_best.pth')
            else:
                episodes_without_improvement += 1
            
            if episode % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Best: {best_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            
            if best_reward >= early_stop_reward or episodes_without_improvement >= patience:
                print(f"Early stopping at episode {episode}")
                break
        
        training_time = (datetime.now() - training_start_time).total_seconds()
        
        # Save training results
        results = {
            'rewards_history': rewards_history,
            'best_reward': float(best_reward),
            'training_time': training_time,
            'episodes_completed': episode + 1,
            'final_epsilon': float(self.epsilon)
        }
        
        with open('training_results.json', 'w') as f:
            json.dump(results, f)
        
        return rewards_history
    
    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

if __name__ == "__main__":
    from snake_env import SnakeEnv
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Initialize environment and agent
    env = SnakeEnv(grid_size=10)
    agent = DQNAgent(env, 
                    learning_rate=1e-3,
                    gamma=0.99,
                    buffer_size=10000,
                    batch_size=64,
                    target_update=100,
                    epsilon_decay=0.97)
    
    # Train the agent
    rewards = agent.train(num_episodes=1000)
    
    # Test the trained agent
    state = env.reset()
    print("\nTesting trained DQN agent:")
    total_reward = 0
    done = False
    steps = 0
    max_test_steps = 100
    
    while not done and steps < max_test_steps:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        steps += 1
    
    print(f"Test finished with total reward: {total_reward:.2f} in {steps} steps")
