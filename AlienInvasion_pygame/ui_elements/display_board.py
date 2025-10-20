import pygame


class Scoreboard:
    """ 统计信息显示牌类 """
    def __init__(self, fontSize, name, value):
        self.name = name
        self.value = value
        self.font = pygame.font.SysFont(None, fontSize)
        self.image = self.font.render(self.name + str(self.value), True, (0, 0, 0))
        self.rect = self.image.get_rect()
        # 初始化位置
        self.rect.y, self.rect.x = 10, 10

    def draw(self, stats, screen):
        self.image = self.font.render(self.name + str(self.value), True, (0, 0, 0))
        screen.blit(self.image, self.rect)
