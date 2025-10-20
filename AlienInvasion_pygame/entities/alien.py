import pygame
from pygame.sprite import Sprite
from entities.bullet import AlienBullet
import random


class Alien(Sprite):
    """ 外星人类 """
    def __init__(self, settings):
        super().__init__()
        self.image = pygame.image.load(settings.alien_image)
        self.image = pygame.transform.scale_by(self.image, 0.23)
        self.rect = self.image.get_rect(center=(0, 0))
        self.centerx = float(self.rect.centerx)
        self.centery = float(self.rect.centery)
        self.speed = random.randint(1, 2) 

    def update(self, settings): # update()为Sprite类内置空方法, 由Group()调用，可以重写
        self.centery  += self.speed * settings.alien_speed_factor
        self.rect.centery = self.centery
    
    def fire(self, alienBullets):
        """ 发射子弹 """
        bullet = AlienBullet(self)
        bullet.rect.center = self.rect.center
        alienBullets.add(bullet)

    def draw(self, screen):
        """ 绘制 """
        screen.blit(self.image, self.rect)
