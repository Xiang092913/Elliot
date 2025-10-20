class GameStats():
    '''统计类'''
    def __init__(self, settings):
        # 初始化
        self.dynamic_initialize(settings)      
    
    def update_score(self, settings, score_board):
        '''更新玩家得分'''
        self.score += settings.alien_point
        self.aliens_killed += 1
        score_board.value = self.score
        
    def level_up(self):
        '''进入下一关游戏'''
        self.stage += 1
        self.aliens_killed = 0
        
    def dynamic_initialize(self, settings):
        '''动态初始化游戏统计数据'''
        self.stage = 1
        self.ship_left = settings.ship_left
        self.score = 0
        self.aliens_killed = 0