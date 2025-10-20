"""
标题动画
"""
import pygame


class GameTitle():
    def __init__(self, screen):
        self.screen = screen
        self.screen_rect = self.screen.get_rect()

        self.font = pygame.font.SysFont(None, 90)
        self.img = self.font.render('Alien Invasion', True, (0, 0, 220), (230, 230, 230))
        self.rect = self.img.get_rect()
        self.reset()

    def reset(self):
        self.rect.x = self.screen_rect.width
        self.rect.centery = self.screen_rect.centery - 200

    def update(self):
        if self.rect.centerx >= self.screen_rect.centerx:
            self.rect.x -= 5
        self.screen.blit(self.img, self.rect)




