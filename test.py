import pygame

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Test Window")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill((0, 0, 255))  # 蓝色背景
    pygame.draw.rect(screen, (255, 0, 0), (100, 100, 50, 50))  # 红色方块
    pygame.display.flip()

pygame.quit()