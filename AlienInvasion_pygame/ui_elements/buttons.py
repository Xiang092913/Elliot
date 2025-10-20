from pygame import font

class Button():
    '''button类'''
    def __init__(self, screen, msg, fontSize):
        self.screen = screen
        self.screen_rect = screen.get_rect()
        self.font = font.SysFont(None, fontSize, italic=True)
        self.image = self.font.render(msg, True, (0, 255, 0), (100, 100, 100))
        self.rect = self.image.get_rect()
   
    def draw(self, centerx, centery):
        self.rect.centerx = centerx
        self.rect.centery = centery
        self.screen.blit(self.image, self.rect)
