import numpy as np
import gym
from gym import spaces

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(SnakeEnv, self).__init__()
        
        self.grid_size = grid_size
        self.reset_state()
        
        # Action space: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        self.action_space = spaces.Discrete(4)
        
        # Observation space: grid_size x grid_size matrix with:
        # 0: empty, 1: snake body, 2: snake head, 3: food
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(self.grid_size, self.grid_size),
            dtype=np.int32
        )
        
        # Reward settings
        self.reward_food = 1.0    # Reward for eating food
        self.reward_move = -0.01  # Small penalty for each move
        self.reward_dead = -1.0   # Penalty for dying
        
    def reset_state(self):
        """Initialize game state"""
        self.snake_pos = [(self.grid_size//2, self.grid_size//2)]  # Snake starts in middle
        self.snake_dir = 1  # Start moving right
        self.food_pos = self._place_food()
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 2
        return self._get_state()
        
    def reset(self):
        """Reset the environment"""
        return self.reset_state()
    
    def _place_food(self):
        """Place food in random empty cell"""
        while True:
            food = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            if food not in self.snake_pos:
                return food
    
    def _get_state(self):
        """Convert game state to observation matrix"""
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place snake body
        for x, y in self.snake_pos[:-1]:
            state[x, y] = 1
            
        # Place snake head
        x, y = self.snake_pos[-1]
        state[x, y] = 2
        
        # Place food
        state[self.food_pos[0], self.food_pos[1]] = 3
        
        return state
    
    def step(self, action):
        """Execute one step in the environment"""
        self.steps += 1
        
        # Get new head position based on action
        head_x, head_y = self.snake_pos[-1]
        if action == 0:  # UP
            new_head = (head_x - 1, head_y)
        elif action == 1:  # RIGHT
            new_head = (head_x, head_y + 1)
        elif action == 2:  # DOWN
            new_head = (head_x + 1, head_y)
        else:  # LEFT
            new_head = (head_x, head_y - 1)
            
        # Check if game is over
        done = False
        reward = self.reward_move
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            return self._get_state(), self.reward_dead, True, {}
        
        # Check self collision
        if new_head in self.snake_pos[:-1]:
            return self._get_state(), self.reward_dead, True, {}
        
        # Move snake
        self.snake_pos.append(new_head)
        
        # Check food collision
        if new_head == self.food_pos:
            reward = self.reward_food
            self.food_pos = self._place_food()
        else:
            self.snake_pos.pop(0)
        
        # Check if max steps reached
        if self.steps >= self.max_steps:
            done = True
            
        return self._get_state(), reward, done, {}
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            state = self._get_state()
            symbols = {0: '.', 1: 'o', 2: 'O', 3: 'X'}
            for row in state:
                print(''.join(symbols[cell] for cell in row))
            print("\n")
