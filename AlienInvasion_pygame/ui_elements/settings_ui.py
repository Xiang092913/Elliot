import pygame


class SettingsUI():
    """ settings用户界面 """
    def __init__(self, screen):
        self.screen = screen
        self.width, self.height =  400, 400
        self.rect = pygame.Rect(0, 0, self.width, self.height)
        self.screen_rect = self.screen.get_rect()
        self.rect.center = self.screen_rect.center
        # 界面标题
        title_font = pygame.font.SysFont('KAITI', 42)
        self.title = title_font.render('游戏设置', True, (0, 0, 0))
        self.title_rect = self.title.get_rect()
        self.title_rect.centerx, self.title_rect.centery = self.rect.centerx, self.rect.y + 50
    
    def draw(self, settings):
        """绘制settings用户界面"""
        pygame.draw.rect(self.screen, color=(180, 180, 180), rect=self.rect)
        pygame.draw.rect(self.screen, color=(90, 90, 90), rect=self.rect, width=4, border_radius=4)  
        self.screen.blit(self.title, self.title_rect)
        # 渲染设置项目
        self.alien_sound_switch = self._draw_ui_item('外星人音效', 
                label_pos=(self.rect.x+50, self.rect.y+150), 
                switch_rect=(self.rect.centerx+50, self.rect.y+150, 90, 30), 
                is_on=settings.alien_sound_on)
        self.ship_sound_switch = self._draw_ui_item( '飞船音效', 
                label_pos=(self.rect.x+50, self.rect.y+200), 
                switch_rect=(self.rect.centerx+50, self.rect.y+200, 90, 30), 
                is_on=settings.ship_sound_on)
        self.bg_music_switch = self._draw_ui_item( '背景音乐', 
                label_pos=(self.rect.x+50, self.rect.y+250), 
                switch_rect=(self.rect.centerx+50, self.rect.y+250, 90, 30), 
                is_on=settings.bg_music_on)
        return self.alien_sound_switch, self.ship_sound_switch, self.bg_music_switch

    def _draw_ui_item(self, text, label_pos, switch_rect, is_on=True):
        """ 绘制ui item """
        # label
        font = pygame.font.SysFont('KAITI', 24)
        label = font.render(text, True, (0, 0, 0))
        self.screen.blit(label, label_pos)
        # 开关按钮矩形
        switch_rect = pygame.draw.rect(self.screen, (190, 190, 190), switch_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), switch_rect, 2)  # 开关按钮边框
        # 绘制开关滑块
        if is_on:
            pygame.draw.rect(self.screen, (0, 240, 0), 
                (switch_rect.x, switch_rect.y, 45, 30))
            text = font.render("ON", True, (0,0,0))
            self.screen.blit(text, (switch_rect.centerx, switch_rect.y))
        else:
            pygame.draw.rect(self.screen, (240, 0, 0), 
                (switch_rect.centerx, switch_rect.y, 45, 30))
            text = font.render("OFF", True, (0,0,0))
            self.screen.blit(text, (switch_rect.x, switch_rect.y))
        return switch_rect
