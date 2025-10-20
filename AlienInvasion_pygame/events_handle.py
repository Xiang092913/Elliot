"""
事件处理模块
"""
import sys, random, time
import pygame

from functions import (create_alien_fleet, is_before_game,
            is_game_over, reload_game, play_sound, stop_sound)

def check_user_events(ctx):
    '''监测玩家user的输入事件'''
    event_list = pygame.event.get()
    for ev in event_list:
        #玩家单击游戏窗口的关闭按钮时，将检测到pygame.QUIT事件
        if ev.type == pygame.QUIT:
            pygame.mixer.music.stop()
            pygame.quit()
            sys.exit()
        # 定时器事件
        if ev.type == ctx.CREATE_ALIEN:
            if ctx.settings.active and not ctx.settings.pause:
                create_alien_fleet(ctx) 
        if ev.type == ctx.ALIEN_FIRE and ctx.settings.active and not ctx.settings.pause:
            visible_aliens = [alien for alien in ctx.aliens.sprites() if alien.rect.top > 0]
            if visible_aliens:
                alien = random.choice(ctx.aliens.sprites())
                alien.fire(ctx.alienBullets)
                if ctx.settings.alien_sound_on:
                    play_sound(ctx.settings.alien_fire_sound)
        # 鼠标输入事件    
        if ev.type == pygame.MOUSEBUTTONDOWN:
            if not ctx.settings.active:   # 开始或game_over界面  
                check_buttons(ctx)
            if ctx.settings.settings_ui_on:  # 检查设置UI界面
                check_settings_ui(ctx)
        if ev.type == pygame.MOUSEMOTION:
            ctx.mouse_motion_start = time.time()
        # 键盘事件
        if ev.type == pygame.KEYDOWN:
            check_keydown_events(ctx, ev)
        if ev.type == pygame.KEYUP:  # 按键抬起
            if ev.key == pygame.K_a: 
                ctx.ship.moving_left = False   
            if ev.key == pygame.K_d: 
                ctx.ship.moving_right = False        
            if ev.key == pygame.K_w: 
                ctx.ship.moving_up = False 
            if ev.key == pygame.K_s:
                ctx.ship.moving_down = False

def check_keydown_events(ctx, ev):
    """ 处理键盘输入事件 """
    if ev.key == pygame.K_SPACE:
        ctx.settings.settings_ui_on = not ctx.settings.settings_ui_on
    if ctx.settings.active:  # 游戏进行中
        if ctx.settings.pause and ev.key == pygame.K_RETURN: 
            reload_game(ctx)  # 暂停中按Enter键重载游戏
        if ev.key == pygame.K_ESCAPE:
            ctx.settings.pause = not ctx.settings.pause
        if ev.key == pygame.K_d:
            ctx.ship.moving_right = True
        if ev.key == pygame.K_a:
            ctx.ship.moving_left = True
        if ev.key == pygame.K_w:
            ctx.ship.moving_up = True
        if ev.key == pygame.K_s:
            ctx.ship.moving_down = True
        if ev.key == pygame.K_l:
            if len(ctx.shipBullets) < ctx.settings.ship_bullets_limit:
                ctx.ship.fire(ctx.shipBullets) # 飞船发射子弹 
                
                if ctx.settings.ship_sound_on:
                    play_sound(ctx.settings.ship_fire_sound)
        if ev.key == pygame.K_k:  
            # print('len(ctx.tripleBullets):', len(ctx.tripleBullets))
            if ctx.stats.score >= 200 and len(ctx.tripleBullets) < ctx.settings.ship_triple_bullets_limit:
                ctx.ship.fire_triple_bullets(ctx.tripleBullets)  
                ctx.stats.score -= 100
                ctx.score_board.value -= 100  
                if ctx.settings.ship_sound_on:
                    play_sound(ctx.settings.ship_fire_sound) 
    else:
        if ev.key == pygame.K_RETURN:
            if is_before_game(ctx):
                ctx.settings.active = True
                if ctx.settings.bg_music_on:
                    pygame.mixer.music.play(loops=-1, start=0)
            if is_game_over(ctx):  # Game Over界面
                reload_game(ctx)            

def check_buttons(ctx):
    '''处理游戏按钮被点击'''
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if is_before_game(ctx):  # Play按钮只能在开始界面有效
        if ctx.play_button.rect.collidepoint(mouse_x, mouse_y): # 开始按钮被点击
            ctx.settings.active = True
            if ctx.settings.bg_music_on:
                pygame.mixer.music.play(-1, 0)
    elif is_game_over(ctx):  # Restart按钮只能在game_over界面有效
        if ctx.restart_button.rect.collidepoint(mouse_x, mouse_y): # Restart按钮被点击, 重置游戏
            reload_game(ctx)

def check_settings_ui(ctx):
    """ 检查设置窗口界面 """
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if ctx.settings_ui.alien_sound_switch.collidepoint(mouse_x, mouse_y):
        ctx.settings.alien_sound_on = not ctx.settings.alien_sound_on
        if not ctx.settings.alien_sound_on:
            stop_sound(ctx.settings.alien_fire_sound)
            stop_sound(ctx.settings.explode_sound)

    if ctx.settings_ui.ship_sound_switch.collidepoint(mouse_x, mouse_y):
        ctx.settings.ship_sound_on = not ctx.settings.ship_sound_on
        if not ctx.settings.ship_sound_on:
            stop_sound(ctx.settings.ship_fire_sound)

    if ctx.settings_ui.bg_music_switch.collidepoint(mouse_x, mouse_y):
        ctx.settings.bg_music_on = not ctx.settings.bg_music_on
        if not ctx.settings.bg_music_on:
            pygame.mixer.music.stop()
        else:
            pygame.mixer.music.play(-1)
