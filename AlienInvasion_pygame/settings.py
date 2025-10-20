

class GameSettings:
    '''设置类'''
    def __init__(self):
        # 游戏屏幕
        self.screen_width = 650
        self.screen_height = 850
        self.bg_color = (230, 230, 230)
        # 图片音效资源
        self.bg_image = r'resources/images/bg3.jpg'
        self.bg_music = r'resources/sounds/血染的战神.mp3'
        self.ship_image = r'resources/images/ship66.png'
        self.alien_image = r'resources/images/alien111.png'
        self.ship_fire_sound = r'resources/sounds/fire.wav'
        self.alien_fire_sound = r'resources/sounds/fire2.wav'
        self.explode_sound = r'resources/sounds/alien_explosion.wav'
        # 动态初始化
        self.dynamic_initialize()
        
    def dynamic_initialize(self):
        self.active = False # 开始、结束标志
        self.pause = False # 暂停开关
        self.settings_ui_on = False  #  设置窗口开关
        self.alien_sound_on = True  #  音效开关
        self.ship_sound_on = True
        self.bg_music_on = True
        # 初始飞船数量
        self.ship_left = 3
        # 飞船、子弹、外星人初始速度
        self.ship_speed = 3.5
        self.alien_bullets_speed = 4.5
        self.ship_bullets_speed = 3.5
        self.alien_point = 20
        self.max_alien_fleet = 4  
        self.level_up_required_aliens = 20  # 升级需要消灭的外星人数量
        self.alien_speed_factor = 1
        
        self.ship_bullets_limit = 8
        self.ship_triple_bullets_limit = 12
    
    def speed_up(self):
        """升级后加快速度"""
        self.ship_speed *= 1.1
        self.ship_bullets_speed *= 1.1
        self.alien_bullets_speed *= 1.1
        self.alien_speed_factor *= 1.1
        self.alien_point += 10
        self.max_alien_fleet += 1
        
