import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
from snake import SnakeGame, Direction

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.vals = []
        self.dones = []
        
    def generate_batch(self):
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        probs = torch.FloatTensor(np.array(self.probs))
        vals = torch.FloatTensor(np.array(self.vals))
        rewards = torch.FloatTensor(np.array(self.rewards))
        dones = torch.FloatTensor(np.array(self.dones))
        
        return states, actions, probs, vals, rewards, dones
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.vals = []
        self.dones = []

class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 0.2
        self.learning_rate = 0.0003
        self.n_epochs = 10
        self.batch_size = 64
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.memory = PPOMemory()
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[0][action].item(), value.item()
    
    def train(self):
        states, actions, old_probs, vals, rewards, dones = self.memory.generate_batch()
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_probs = old_probs.to(self.device)
        vals = vals.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # 计算优势
        advantages = []
        advantage = 0
        for r, v, done in zip(reversed(rewards), reversed(vals), reversed(dones)):
            if done:
                advantage = 0
            advantage = r + self.gamma * advantage * (1 - done) - v
            advantages.insert(0, advantage)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            for idx in range(0, len(states), self.batch_size):
                batch_states = states[idx:idx+self.batch_size]
                batch_actions = actions[idx:idx+self.batch_size]
                batch_old_probs = old_probs[idx:idx+self.batch_size]
                batch_advantages = advantages[idx:idx+self.batch_size]
                
                probs, values = self.policy(batch_states)
                probs = probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
                
                ratio = probs / batch_old_probs
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = nn.MSELoss()(values.squeeze(), rewards[idx:idx+self.batch_size])
                
                loss = actor_loss + 0.5 * critic_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        self.memory.clear_memory()

def train_ppo(episodes=1000):
    env = SnakeGame()
    state_size = 8
    action_size = 4
    agent = PPOAgent(state_size, action_size)
    
    scores = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, prob, val = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.states.append(state)
            agent.memory.actions.append(action)
            agent.memory.probs.append(prob)
            agent.memory.vals.append(val)
            agent.memory.rewards.append(reward)
            agent.memory.dones.append(done)
            
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
    torch.save(agent.policy.state_dict(), "ppo_model.pth")
    print("训练完成，模型已保存。")

def test_ppo(model_path="ppo_model.pth"):
    env = SnakeGame()
    state_size = 8
    action_size = 4
    agent = PPOAgent(state_size, action_size)
    agent.policy.load_state_dict(torch.load(model_path))
    agent.policy.eval()
    
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        action, _, _ = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
    
    print(f"测试结束，得分: {total_reward}")
    pygame.time.wait(2000)

if __name__ == "__main__":
    train_ppo()
    # test_ppo() 