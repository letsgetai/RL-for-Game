import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
from snake import SnakeGame, Direction

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ACAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        self.memory = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[0][action].item()
    
    def train(self):
        if not self.memory:
            return
            
        states, actions, rewards, next_states, dones = zip(*self.memory)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算TD误差
        current_values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        td_targets = rewards + (1 - dones) * self.gamma * next_values
        td_errors = td_targets - current_values
        
        # 更新Critic
        critic_loss = nn.MSELoss()(current_values, td_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        probs = self.actor(states)
        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
        actor_loss = -(torch.log(action_probs) * td_errors.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.memory = []

def train_ac(episodes=1000):
    env = SnakeGame()
    state_size = 8
    action_size = 4
    agent = ACAgent(state_size, action_size)
    
    scores = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.train()
        scores.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode+1}, Average Score: {avg_score:.2f}")
    
    # 保存模型
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict()
    }, "ac_model.pth")
    print("训练完成，模型已保存。")

def test_ac(model_path="ac_model.pth"):
    env = SnakeGame()
    state_size = 8
    action_size = 4
    agent = ACAgent(state_size, action_size)
    
    checkpoint = torch.load(model_path)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.actor.eval()
    agent.critic.eval()
    
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        action, _ = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
    
    print(f"测试结束，得分: {total_reward}")
    pygame.time.wait(2000)

if __name__ == "__main__":
    train_ac()
    # test_ac() 