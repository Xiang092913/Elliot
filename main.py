"""
主模块
"""
import time
import pygame

from context import Context
import functions as funcs
from events_handle import check_user_events

class Game:
    def __init__(self):
        #初始化pygame
        pygame.init()  
        pygame.mixer.init()
        pygame.display.set_caption('外星人入侵')
        self.ctx = Context()
        self.clock = pygame.time.Clock()  # 时钟
        pygame.time.set_timer(self.ctx.CREATE_ALIEN, 5000)  # 定时器，每5000ms创建一次外星人群
        pygame.time.set_timer(self.ctx.ALIEN_FIRE, 3000)  # 定时器，外星人每3000ms发射一次子弹

    def run(self):
        """ 运行游戏循环 """
        while True:   
            self.clock.tick(60)  # 每秒60帧
            check_user_events(self.ctx)  # 监测用户事件 
            if self.ctx.settings.active and not self.ctx.settings.pause: # 游戏在运行状态下，更新所有角色
                self.ctx.ship.move(self.ctx.settings)  # 飞船船移动位置
                funcs.update_fleet(self.ctx) # 更新外星人位置
                funcs.update_bullets(self.ctx) # 更新子弹位置
                funcs.check_bullet_alien_collision(self.ctx, self.ctx.shipBullets)
                funcs.check_bullet_alien_collision(self.ctx, self.ctx.tripleBullets) 
                funcs.check_ship_collision(self.ctx)
                funcs.update_particles(self.ctx)   # 爆炸颗粒效果

            if time.time() - self.ctx.mouse_motion_start > 2:  # 鼠标光标静止超过2秒后隐藏鼠标
                pygame.mouse.set_visible(False)
            else:
                pygame.mouse.set_visible(True)
            funcs.update_screen(self.ctx)  # 更新屏幕


if __name__ == '__main__':
    game = Game()
    game.run()
