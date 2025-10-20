"""
函数模块：定义游戏中使用的函数
"""
import time, random 
import pygame
from context import Context
from entities.alien import Alien
from entities.particles import Particle

def create_alien_fleet(ctx):
    '''创建外星人舰队'''
    for i in range(2, ctx.settings.max_alien_fleet):
        alien = Alien(ctx.settings)
        space_y = int(alien.rect.height + 50)
        alien.rect.x = random.randint(0, ctx.screen_rect.width - alien.rect.width)
        alien.centery = random.choice([0, space_y, space_y * 2])
        ctx.aliens.add(alien)
    
def update_fleet(ctx):
    '''更新外星舰队'''
    ctx.aliens.update(ctx.settings)

def update_particles(ctx):
    '''更新外星人爆炸粒子'''
    ctx.particles.update()   

def update_bullets(ctx):
    '''更新子弹位置和数量'''
    for bullets in [ctx.shipBullets, ctx.tripleBullets, ctx.alienBullets]:
        bullets.update(ctx.settings)
    remove_overflow_bullets(ctx) 

def remove_overflow_bullets(ctx):
    '''当子弹不在屏幕区域时, 删除子弹'''
    for bullet in ctx.shipBullets.copy():
        if bullet.rect.bottom <= 0:
            ctx.shipBullets.remove(bullet)
    for bullet in ctx.tripleBullets.copy():
        if bullet.rect.bottom <= 0:
            ctx.tripleBullets.remove(bullet)
    for bullet in ctx.alienBullets.copy():
        if bullet.rect.bottom >= ctx.screen_rect.bottom:
            ctx.alienBullets.remove(bullet)
            
def check_bullet_alien_collision(ctx, bullets):
    '''如果bullet和alien碰撞'''
    collisions = pygame.sprite.groupcollide(bullets, ctx.aliens, True, True)  # pygame自动删除发生碰撞的精灵
    if collisions:
        for alien_list in collisions.values():
            for alien in alien_list:  # 一次爆炸可能有多个alien
                ctx.stats.score += ctx.settings.alien_point
                ctx.stats.aliens_killed += 1
                ctx.score_board.value = ctx.stats.score  # 更新记分牌
                alien_explode(ctx, alien)

                if ctx.settings.alien_sound_on:
                    play_sound(ctx.settings.explode_sound)

        if ctx.stats.aliens_killed >= ctx.settings.level_up_required_aliens + 5 * ctx.stats.stage:
            reset_sprites(ctx)
            ctx.stats.level_up() # 进入下一关
            ctx.settings.speed_up()
            ctx.stage_board.value = ctx.stats.stage  # 更新关卡

def alien_explode(ctx, alien):
    """外星人爆炸"""
    if isinstance(alien, Alien):  # alien是Alien类实例
        for _ in range(150):  # 爆炸产生的粒子数量150
            particle = Particle(alien.rect.centerx, alien.rect.centery)
            ctx.particles.add(particle)
                   
def check_ship_collision(ctx):
    '''检查alien和ship相撞'''
    if pygame.sprite.spritecollideany(ctx.ship, ctx.aliens) or \
        alien_reach_screen_bottom(ctx) or \
        pygame.sprite.spritecollideany(ctx.ship, ctx.alienBullets):
        
        ctx.stats.ship_left -= 1  # 减少飞船数量
        ctx.ships_left_board.value = ctx.stats.ship_left  # 更新显示牌
        if ctx.stats.ship_left > 0:          
            reset_sprites(ctx)
        else:
            ctx.settings.active = False
            pygame.mixer.music.stop()
        time.sleep(1)  # 暂停1秒
        
def alien_reach_screen_bottom(ctx):
    '''外星人到达屏幕底部'''
    for alien in ctx.aliens.sprites():
        if alien.rect.bottom >= ctx.screen_rect.bottom:
            return True

def reset_sprites(ctx):
    '''重置游戏精灵'''
    ctx.aliens.empty() # 清空aliens group
    ctx.particles.empty()
    ctx.shipBullets.empty() # 清空bullets group
    ctx.tripleBullets.empty()
    ctx.alienBullets.empty()
    ctx.ship.set_center()  

def reload_game(ctx: Context):
    '''重载游戏'''
    # 重置精灵
    reset_sprites(ctx)
    # 动态初始化设置和统计信息
    ctx.settings.dynamic_initialize()
    ctx.stats.dynamic_initialize(ctx.settings)
    # 记分牌重置
    ctx.score_board.value = ctx.stats.score
    ctx.stage_board.value = ctx.stats.stage
    ctx.ships_left_board.value = ctx.stats.ship_left
    # 重置标题
    ctx.game_title.reset() 
   
    if ctx.settings.bg_music_on:
        pygame.mixer.music.stop()

def update_screen(ctx):
    '''更新屏幕'''
    ctx.screen.fill(ctx.settings.bg_color)
    if ctx.settings.active:
        # ctx.screen.blit(ctx.background, ctx.screen_rect)  # 背景
        for board in [ctx.score_board, ctx.stage_board, ctx.ships_left_board]:  # 绘制记分牌
            board.draw(ctx.stats, ctx.screen)
        ctx.ship.draw(ctx.screen)  # 绘制飞船
        for group in [ctx.aliens, ctx.particles, ctx.shipBullets, ctx.tripleBullets, 
                            ctx.alienBullets]:  # 绘制所有精灵
            for sprite in group.sprites():  
                sprite.draw(ctx.screen)

    elif is_before_game(ctx):  # 显示标题动画
        ctx.game_title.update()
        if ctx.game_title.rect.centerx <= ctx.screen_rect.centerx:
            ctx.play_button.draw(ctx.screen_rect.centerx, ctx.screen_rect.centery)
    elif is_game_over(ctx):  # 结束后，绘制Restart按钮

        ctx.restart_button.draw(ctx.screen_rect.centerx, ctx.screen_rect.centery + 200) 

    if ctx.settings.settings_ui_on:  # 显示设置窗口
        ctx.settings_ui.draw(ctx.settings) 
    pygame.display.update()   # 更新屏幕

def play_sound(sound):
    """ 播放音效 """
    s = pygame.mixer.Sound(sound)
    s.play(0)

def stop_sound(sound):
    """关闭音效"""
    s = pygame.mixer.Sound(sound)
    s.stop()

def is_before_game(ctx):
    """ 判断是否在开始游戏前界面 """
    return ctx.stats.ship_left == ctx.settings.ship_left

def is_game_over(ctx):
    """ 判断是否在游戏结束界面 """
    return ctx.stats.ship_left <= 0
