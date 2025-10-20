import pygame
from pygame.sprite import Sprite

class Bullet(Sprite):
    """ 子弹类， 继承pygame的Sprite类 """
    def __init__(self, ship):  # 继承sprite模块的Sprite类，通过使用精灵，可将游戏中的元素编组，进而同时操作编组中的所有元素
        super().__init__() 
        self.width = 6
        self.height = 10
        self.color = (60, 60, 60)
        self.rect = pygame.Rect(0, 0, self.width, self.height)
        # 子弹初始位置
        self.rect.centerx = ship.rect.centerx
        self.rect.top = ship.rect.top
        # 子弹y坐标存为float小数类型
        self.y = float(self.rect.y)
        
    def update(self, settings):
        """ 更新子弹位置 """
        self.y -= settings.ship_bullets_speed
        self.rect.y = self.y

    def draw(self, screen):
        """ 绘制 """
        pygame.draw.rect(screen, self.color, self.rect) # pygame.draw.rect()


class AlienBullet(Bullet):
    """ 外星人子弹类，继承子弹类 """
    def __init__(self, alien):
        super().__init__(alien)
        self.width = 12
        self.height = 25
        self.color = (180, 40, 240)
        self.rect = pygame.Rect(0, 0, self.width, self.height)
        self.rect.centerx = alien.rect.centerx
        self.rect.top = alien.rect.bottom 
        self.y = float(self.rect.y)

    def update(self, settings):
        """ 更新子弹位置 """
        self.y += settings.alien_bullets_speed
        self.rect.y = self.y