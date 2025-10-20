import pygame
from entities.bullet import Bullet
from pygame import transform


class Ship:
    """ 飞船类 """
    def __init__(self, screen, settings):
        self.img = pygame.image.load(settings.ship_image)
        self.image = transform.scale_by(self.img, 0.15)
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        # 初始位置
        self.rect.bottom = self.screen_rect.bottom
        self.rect.centerx = self.screen_rect.centerx
        self.bottom = float(self.rect.bottom)  # 存储位置浮点数，更精确移动
        self.centerx = float(self.rect.centerx)

        self.moving_right = False
        self.moving_left = False
        self.moving_up = False
        self.moving_down = False

    def fire(self, shipBullets):
        '''发射子弹'''
        new_bullet = Bullet(self)
        shipBullets.add(new_bullet)

    def fire_triple_bullets(self, tripleBullets):
        '''发射散弹'''
        b1 = Bullet(self)
        b1.rect.left = self.rect.left - 30
        b2 = Bullet(self)
        b2.rect.centerx = self.rect.centerx
        b3 = Bullet(self)
        b3.rect.right = self.rect.right + 30
        for b in (b1, b2, b3):
            tripleBullets.add(b)
 

    def move(self, settings):
        """ 飞船移动 """
        if self.moving_right and self.rect.right < self.screen_rect.right:
            self.centerx += settings.ship_speed
        if self.moving_left and self.rect.left > 0:
            self.centerx -= settings.ship_speed
        if self.moving_up and self.rect.top > 0:
            self.bottom -= settings.ship_speed
        if self.moving_down and self.rect.bottom < self.screen_rect.bottom:
            self.bottom += settings.ship_speed

        self.rect.centerx = self.centerx
        self.rect.bottom = self.bottom

    def set_center(self):
        """ 将飞船居中 """
        self.centerx = self.screen_rect.centerx
        self.bottom = self.screen_rect.bottom
        self.rect.centerx = self.centerx
        self.rect.bottom = self.bottom
    
    def draw(self, screen):
        """ 绘制飞船 """
        screen.blit(self.image, self.rect)
