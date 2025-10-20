import math, random
import pygame


class Particle(pygame.sprite.Sprite):
    """外星人爆炸颗粒"""
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.color = (15, 40, 15)
        self.radius = random.randint(2, 10)
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = random.uniform(1, 3)
        self.gravity = 0.6
 
    def update(self):
        self.x += math.sin(self.angle) * self.speed
        self.y += math.cos(self.angle) * self.speed + self.gravity
        self.radius -= 1  # 粒子逐渐变小
 
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.radius))