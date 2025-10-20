"""
游戏运行使用的上下文变量
"""
import time
import pygame
from pygame import display, image, sprite, mixer, transform

from entities.ship import Ship
from ui_elements.title_animation import GameTitle
from ui_elements.display_board import Scoreboard
from ui_elements.buttons import Button
from ui_elements.settings_ui import SettingsUI
from settings import GameSettings
from stats import GameStats


class Context:
    def __init__(self):
        self.settings = GameSettings() # 游戏设置初始化（创建设置类实例）
        self.stats = GameStats(self.settings) # 统计
        # 创建屏幕surface和caption，加载背景图像
        self.screen = display.set_mode((self.settings.screen_width, self.settings.screen_height))  
        self.screen_rect = self.screen.get_rect()
        self.background = image.load(self.settings.bg_image)
        self.background = transform.scale(self.background, self.screen_rect.size)
        self.bg_music = mixer.music.load(self.settings.bg_music)  # 加载背景音乐
        # 创建游戏中的元素：按钮，记分牌，飞船，外星人，子弹
        # 按钮
        self.play_button = Button(self.screen, 'Play', 60)
        self.restart_button = Button(self.screen, 'Game over! Click to Start Again', 40)
        self.game_title = GameTitle(self.screen)  # 游戏标题动画
        self.settings_ui = SettingsUI(self.screen)  # 设置界面
        # 记分牌（显示游戏当前得分、等级、剩余飞船数量）
        self.score_board = Scoreboard(22, 'Score:', self.stats.score)
        self.stage_board = Scoreboard(22, 'Stage:', self.stats.stage)
        self.stage_board.rect.centerx = self.screen_rect.centerx  # 位置
        self.ships_left_board = Scoreboard(22, 'Ships_left: ', self.stats.ship_left)
        self.ships_left_board.rect.right = self.screen_rect.width - 10  # 位置
        # 使用sprite模块, 创建子弹、外星人的空编组，用来批量储存、操作精灵
        self.shipBullets = sprite.Group() 
        self.tripleBullets = sprite.Group()  # 三连发子弹编组
        self.alienBullets = sprite.Group()
        self.aliens = sprite.Group()
        self.particles  = sprite.Group()
        # 创建飞船
        self.ship = Ship(self.screen, self.settings)  
        # 自定义事件
        self.CREATE_ALIEN = pygame.USEREVENT 
        self.ALIEN_FIRE = pygame.USEREVENT + 1
        self.mouse_motion_start = time.time()