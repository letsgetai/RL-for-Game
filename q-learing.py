import numpy as np
import random
from snake import SnakeGame, Direction
import time
import pickle
import pygame
# 初始化Q表和超参数
Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.3
min_epsilon = 0.01
epsilon_decay = 0.995

def get_state_key(game):
    state = game._get_state()
    return str(state.tolist())

def init_state(state):
    if state not in Q:
        Q[state] = np.zeros(4)

def choose_action(state):
    init_state(state)
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(Q[state])

def save_qtable(filename='qtable.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(Q, f)

def load_qtable(filename='qtable.pkl'):
    global Q
    try:
        with open(filename, 'rb') as f:
            Q = pickle.load(f)
    except FileNotFoundError:
        Q = {}

def train():
    global epsilon
    load_qtable()  # 加载已有的Q表
    
    for episode in range(200):
        game = SnakeGame(speed=20)  # 较慢的训练速度
        state = get_state_key(game)
        init_state(state)
        total_reward = 0
        
        while True:
            game.clock.tick(20)  # 控制训练速度
            
            action = choose_action(state)
            _, reward, done, _ = game.step(Direction(action))
            next_state = get_state_key(game)
            init_state(next_state)
            total_reward += reward
            
            # Q-learning更新
            best_next_q = np.max(Q[next_state])
            current_q = Q[state][action]
            Q[state][action] = current_q + alpha * (reward + gamma * best_next_q - current_q)
            
            if done:
                if episode % 50 == 0:
                    print(f"Episode {episode}, Score: {game.score}, Epsilon: {epsilon:.3f}")
                    save_qtable()  # 定期保存
                break
                
            state = next_state
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    save_qtable()  # 训练完成后保存

def test():
    load_qtable()
    game = SnakeGame(speed=10)
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        state = get_state_key(game)
        action = np.argmax(Q[state]) if state in Q else 0
        _, _, done, _ = game.step(Direction(action))
        game.render()
        
        if done:
            print(f"Test Score: {game.score}")
            game.reset()
        
        game.clock.tick(10)

if __name__ == "__main__":
    train()
    test()