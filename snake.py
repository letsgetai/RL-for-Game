import pygame
import numpy as np
import random
from enum import Enum
from typing import Tuple

# 方向枚举（离散动作）
class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class SnakeGame:
    def __init__(self, width: int = 640, height: int = 480, block_size: int = 20, speed: int = 10):
        """初始化游戏参数"""
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed  # 控制蛇的移动速度（值越小越快）
        
        # PyGame初始化
        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        
        # 游戏状态初始化
        self.reset()

    def reset(self) -> np.ndarray:
        """重置游戏状态"""
        self.direction = Direction.RIGHT
        self.head = [self.width // 2, self.height // 2]
        self.snake = [
            self.head.copy(),
            [self.head[0] - self.block_size, self.head[1]],
            [self.head[0] - 2 * self.block_size, self.head[1]]
        ]
        self.score = 0
        self.food = self._place_food()
        self.last_move_time = pygame.time.get_ticks()
        return self._get_state()

    def _place_food(self) -> Tuple[int, int]:
        """随机生成食物位置"""
        x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
        return (x, y) if [x, y] not in self.snake else self._place_food()

    def _get_state(self) -> np.ndarray:
        """获取当前状态（简化版）"""
        head_x, head_y = self.head
        food_x, food_y = self.food
        return np.array([
            # 危险方向（简化为四个方向）
            (head_y - self.block_size < 0) or ([head_x, head_y - self.block_size] in self.snake),  # 上
            (head_y + self.block_size >= self.height) or ([head_x, head_y + self.block_size] in self.snake),  # 下
            (head_x - self.block_size < 0) or ([head_x - self.block_size, head_y] in self.snake),  # 左
            (head_x + self.block_size >= self.width) or ([head_x + self.block_size, head_y] in self.snake),  # 右
            # 食物相对方向
            food_x < head_x,  # 左
            food_x > head_x,  # 右
            food_y < head_y,  # 上
            food_y > head_y   # 下
        ], dtype=np.float32)

    def step(self, action: int = None) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作（action为None时保持原方向）"""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_move_time < 1000 // self.speed:  # 控制移动间隔
            return self._get_state(), 0, False, {}

        # 1. 处理方向变化（如果提供了新动作）
        if action is not None:
            if action == Direction.UP.value and self.direction != Direction.DOWN:
                self.direction = Direction.UP
            elif action == Direction.DOWN.value and self.direction != Direction.UP:
                self.direction = Direction.DOWN
            elif action == Direction.LEFT.value and self.direction != Direction.RIGHT:
                self.direction = Direction.LEFT
            elif action == Direction.RIGHT.value and self.direction != Direction.LEFT:
                self.direction = Direction.RIGHT

        # 2. 移动蛇头
        if self.direction == Direction.UP:
            self.head[1] -= self.block_size
        elif self.direction == Direction.DOWN:
            self.head[1] += self.block_size
        elif self.direction == Direction.LEFT:
            self.head[0] -= self.block_size
        elif self.direction == Direction.RIGHT:
            self.head[0] += self.block_size

        self.last_move_time = current_time

        # 3. 检查碰撞
        if (self.head[0] < 0 or self.head[0] >= self.width or
            self.head[1] < 0 or self.head[1] >= self.height or
            self.head in self.snake[1:]):
            return self._get_state(), -10, True, {}

        # 4. 检查食物
        reward = -0.1  # 默认每步小惩罚
        if self.head == list(self.food):
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()

        # 5. 更新蛇身
        self.snake.insert(0, self.head.copy())
        return self._get_state(), reward, False, {}

    def render(self):
        """渲染游戏界面"""
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt[0], pt[1], self.block_size, self.block_size))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        
        font = pygame.font.SysFont('arial', 20)
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [5, 5])
        
        pygame.display.flip()
        self.clock.tick(60)  # 固定渲染帧率

def human_play():
    game = SnakeGame(speed=10)  # 数值越大蛇移动越慢
    running = True
    current_direction = None
    
    while running:
        # 1. 处理输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    current_direction = Direction.UP.value
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    current_direction = Direction.DOWN.value
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    current_direction = Direction.LEFT.value
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    current_direction = Direction.RIGHT.value
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # 2. 游戏逻辑更新
        state, reward, done, _ = game.step(current_direction)
        if done:
            print(f"Game Over! Score: {game.score}")
            game.reset()
            current_direction = None
        
        # 3. 渲染
        game.render()
    
    pygame.quit()

if __name__ == "__main__":
    human_play()