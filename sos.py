import pyglet
from pyglet import shapes, font
from pyglet.window import key
from enum import Enum
import random
import math
import torch
import numpy as np
import os
import sys

# Import Bot Logic
from greedy_bot import SOSBot


# BOT WRAPPER (From sos_bot.py)

class AlphaBot:
    def __init__(self, model_path='checkpoints/best.pth'):
        print("AlphaBot is currently a placeholder.")

    def choose_move_wrapper(self, board, current_sos, current_scores, current_player_idx):
        return None


# UI UTILS & BUTTON CLASS


class Button:
    def __init__(self, x, y, width, height, text, batch, group, img_normal, callback=None, font_size=18, overlay=None, is_selected=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.callback = callback
        self.is_selected = is_selected
        
        # Scale based on image size
        self.base_scale_x = width / img_normal.width
        self.base_scale_y = height / img_normal.height
        
        self.sprite = pyglet.sprite.Sprite(img_normal, x=x, y=y, batch=batch, group=group)
        self.sprite.scale_x = self.base_scale_x
        self.sprite.scale_y = self.base_scale_y
        
        self.img_normal = img_normal
        
        self.overlay_sprite = None
        if overlay:
             self.overlay_sprite = pyglet.sprite.Sprite(overlay, x=x, y=y, batch=batch, group=pyglet.graphics.Group(order=group.order+1))
             self.overlay_sprite.scale_x = width / overlay.width
             self.overlay_sprite.scale_y = height / overlay.height
             
        self.total_time = 0.0

        # Center text
        self.label = None
        if text:
            self.label = pyglet.text.Label(text, font_name=custom_font, font_size=font_size,
                                           x=x + width//2, y=y + height//2,
                                           anchor_x='center', anchor_y='center',
                                           color=(255, 255, 255, 255),
                                           batch=batch, group=pyglet.graphics.Group(order=group.order+2))

    def update(self, dt):
        self.total_time += dt
        
        # Pulse animation for selected state
        base_anim = 1.05 + 0.05 * math.sin(self.total_time * 5.0) if self.is_selected else 1.0
            
        hover_anim = 1.1 if hasattr(self, 'is_hovered') and self.is_hovered else 1.0
        
        final_scale = 1.0
        if self.is_selected:
            final_scale = base_anim
            if hasattr(self, 'is_hovered') and self.is_hovered:
                final_scale += 0.05 
        else:
            final_scale = hover_anim

        # Apply Scale
        final_sx = self.base_scale_x * final_scale
        final_sy = self.base_scale_y * final_scale
        
        self.sprite.scale_x = final_sx
        self.sprite.scale_y = final_sy
        
        # Adjust position to center
        current_w = self.img_normal.width * final_sx
        current_h = self.img_normal.height * final_sy
        
        x_shift = (current_w - self.width) / 2
        y_shift = (current_h - self.height) / 2
        
        self.sprite.x = self.x - x_shift
        self.sprite.y = self.y - y_shift
        
        if self.overlay_sprite:
             o_base_sx = self.width / self.overlay_sprite.image.width
             o_base_sy = self.height / self.overlay_sprite.image.height
             
             self.overlay_sprite.scale_x = o_base_sx * final_scale
             self.overlay_sprite.scale_y = o_base_sy * final_scale
             self.overlay_sprite.x = self.x - x_shift
             self.overlay_sprite.y = self.y - y_shift
        
    def check_hit(self, x, y):
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def check_hover(self, x, y):
        self.is_hovered = self.check_hit(x, y)
        return self.is_hovered

    def on_mouse_press(self, x, y, button, modifiers):
        if self.check_hit(x, y):
            return True
        return False

    def on_mouse_release(self, x, y, button, modifiers):
        if self.check_hit(x, y) and self.callback:
            self.callback()
            return True
        return False

    def delete(self):
        if self.sprite: self.sprite.delete()
        if self.overlay_sprite: self.overlay_sprite.delete()
        if self.label: self.label.delete()


# GAME CONSTANTS & STATE


class GameState(Enum):
    HOME = 0
    SETTINGS_POPUP = 1 
    PLAYING = 2
    GAME_OVER = 3

class GameMode(Enum):
    PVP = 0
    PVE = 1

class BotType(Enum):
    GREEDY = 0
    ALPHA = 1

# Window Setup
window = pyglet.window.Window(fullscreen=True, caption="SOS Game - Ultimate")
pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)

# Global State
game_state = GameState.HOME
game_mode = GameMode.PVP
bot_type = BotType.GREEDY
wrap_around = True

players = ['P1', 'P2'] 
scores = {'P1': 0, 'P2': 0}
current_player = 0 
board = []
SOS = []
hc = None 
selected_cell = None
winner_text = ""
last_move_pos = None # Stores (r, c)
last_move_sprite = None

greedy_bot = SOSBot(wrap_around=True)
alpha_bot = AlphaBot()


# RESOURCES & BATCHES


pyglet.resource.path = ['assets']
pyglet.resource.reindex()

# Load Font
font.add_file('assets/PressStart2P-Regular.ttf')
custom_font = 'Press Start 2P'

# Images
bg_img = pyglet.resource.image('background.png')
title_img = pyglet.resource.image('title.png')
btn_img = pyglet.resource.image('button.png')

pvp_img = pyglet.resource.image('pvp.png')
pve_img = pyglet.resource.image('pve.png')
exit_img = pyglet.resource.image('exit.png') 
greedy_img = pyglet.resource.image('greedy.png')
alphago_img = pyglet.resource.image('alphago.png')
orbit_img = pyglet.resource.image('orbit.png')
start_img = pyglet.resource.image('start.png')
back_img = pyglet.resource.image('back.png') 

cell_bg = pyglet.resource.image('cell.png')
cell_sel = pyglet.resource.image('cell_selected.png')
last_move_img = pyglet.resource.image('last_move.png')
img_s = pyglet.resource.image('s.png')
img_o = pyglet.resource.image('o.png')

# Batches
main_batch = pyglet.graphics.Batch()
bg_group = pyglet.graphics.Group(order=0)
board_back_group = pyglet.graphics.Group(order=1)
highlight_group = pyglet.graphics.Group(order=2)
board_main_group = pyglet.graphics.Group(order=3) 
ui_group = pyglet.graphics.Group(order=4)         
text_group = pyglet.graphics.Group(order=6)

# Board Dimensions
no_of_cells = 8 
line_thickness = 5
space = 5
available_dim = min(window.width, window.height) * 0.9
total_gaps = (no_of_cells - 1) * space
grid_size = (available_dim - total_gaps) / no_of_cells
total_grid_dim = no_of_cells * grid_size + total_gaps
cellSize = no_of_cells * grid_size 

gridX = window.width // 2 - total_grid_dim // 2
gridY = window.height // 2 - total_grid_dim // 2

# Collections
sprites = []     
grid_sprites = [] 
lines = []       
buttons = []     
scheduled_functions = [] 
orbit_dots = []
active_lines = []
history_lines = []

# UI SCENES


import time # Needed for animation

def clear_ui():
    buttons.clear()

def update_ui(dt):
    for b in buttons:
        b.update(dt)
        
def update_effects(dt):
    if game_state != GameState.PLAYING and game_state != GameState.GAME_OVER: return
    
    t = pyglet.clock.tick()
    if not hasattr(update_effects, 'total_time'): update_effects.total_time = 0
    update_effects.total_time += dt
    
    # Pulse active lines (Opacity 150-250, Width Line+0..3)
    alpha = 200 + 55 * math.sin(update_effects.total_time * 5.0)
    thick_add = 1.5 + 1.5 * math.sin(update_effects.total_time * 5.0)
    
    for l in active_lines:
        l.opacity = int(alpha)
        l.width = line_thickness + thick_add

def recalc_history_opacity():
    # Gradual fade for older lines (Oldest -> Newest)
    
    total = len(history_lines)
    if total == 0: return
    
    min_op = 40
    max_op = 160
    
    for i, group in enumerate(history_lines):
        # Fraction 0..1
        frac = (i + 1) / total
        op = min_op + (max_op - min_op) * frac
        
        for l in group:
            l.opacity = int(op)
            l.width = line_thickness # Reset Pulse
            
def setup_home():
    clear_ui()
    
    # Title
    t_w = 600
    scale = t_w / title_img.width
    t_h = title_img.height * scale
    
    title_sprite = pyglet.sprite.Sprite(title_img, x=window.width//2 - t_w//2, y=window.height - 380, batch=main_batch, group=ui_group)
    title_sprite.scale = scale
    sprites.append(title_sprite)
    
    # Center logic for animation
    title_sprite.base_scale = scale
    title_sprite.base_x = title_sprite.x
    title_sprite.base_y = title_sprite.y
    title_sprite.center_x = title_sprite.x + t_w / 2
    title_sprite.center_y = title_sprite.y + t_h / 2
    
    def update_title(dt):
        if game_state != GameState.HOME: return
        t = pyglet.clock.tick()
        if not hasattr(update_title, 'total_time'): update_title.total_time = 0
        update_title.total_time += dt
        
        # Breathing animation
        factor = 1.0 + 0.01 * math.sin(update_title.total_time * 2.0)
        new_scale = title_sprite.base_scale * factor
        
        title_sprite.scale = new_scale
        
        current_w = title_img.width * new_scale
        current_h = title_img.height * new_scale
        title_sprite.x = title_sprite.center_x - current_w / 2
        title_sprite.y = title_sprite.center_y - current_h / 2

    pyglet.clock.schedule_interval(update_title, 1/60.0)
    scheduled_functions.append(update_title)

    # 1 vs 1 Button
    btn_w, btn_h = 300, 80
    cx = window.width // 2
    cy = window.height // 2 + 50 # Shift up
    
    b1 = Button(cx - btn_w//2, cy, btn_w, btn_h, "", main_batch, ui_group, btn_img, show_pvp_settings, overlay=pvp_img)
    buttons.append(b1)
    
    # 1 vs Bot Button (Increased gap to 120)
    b2 = Button(cx - btn_w//2, cy - 120, btn_w, btn_h, "", main_batch, ui_group, btn_img, show_bot_settings, overlay=pve_img)
    buttons.append(b2)
    
    # Exit
    b3 = Button(cx - btn_w//2, cy - 240, btn_w, btn_h, "", main_batch, ui_group, btn_img, window.close, overlay=exit_img)
    buttons.append(b3)

def setup_pvp_settings():
    clear_ui()
    py = window.height//2 - 250
    
    # Title
    lbl = pyglet.text.Label("PVP Settings", font_name=custom_font, font_size=36,
                            x=window.width//2, y=py + 500 - 60, anchor_x='center', batch=main_batch, group=text_group)
    sprites.append(lbl)

    # Wrap Toggle
    def toggle_wrap():
        global wrap_around
        wrap_around = not wrap_around
        setup_pvp_settings()
        
    btn_w = 200
    bg_w = btn_img
    b_wrap = Button(window.width//2 - btn_w//2, py + 250, btn_w, 60, "", main_batch, text_group, bg_w, toggle_wrap, overlay=orbit_img, is_selected=wrap_around)
    buttons.append(b_wrap)
    
    # Start
    b_start = Button(window.width//2 - btn_w//2, py + 120, btn_w, 80, "", main_batch, text_group, btn_img, start_pvp, overlay=start_img)
    buttons.append(b_start)
    
    # Back
    b_back = Button(window.width//2 - btn_w//2, py - 60, btn_w, 50, "", main_batch, text_group, btn_img, return_home, overlay=back_img)
    buttons.append(b_back)

def setup_bot_settings():
    clear_ui()
    py = window.height//2 - 250
    
    # Title
    lbl = pyglet.text.Label("Bot Settings", font_name=custom_font, font_size=36,
                            x=window.width//2, y=py + 500 - 60, anchor_x='center', batch=main_batch, group=text_group)
    sprites.append(lbl) 
    
    # Bot Type 
    def set_greedy():
        global bot_type
        bot_type = BotType.GREEDY
        setup_bot_settings() 
        
    def set_alpha():
        global bot_type
        bot_type = BotType.ALPHA
        setup_bot_settings()

    btn_w = 200
    bx = window.width//2 - btn_w - 20
    by = py + 300
    
    bg_g = btn_img
    bg_a = btn_img
    
    b_greedy = Button(bx, by, btn_w, 60, "", main_batch, text_group, bg_g, set_greedy, overlay=greedy_img, is_selected=(bot_type == BotType.GREEDY))
    b_alpha = Button(window.width//2 + 20, by, btn_w, 60, "", main_batch, text_group, bg_a, set_alpha, overlay=alphago_img, is_selected=(bot_type == BotType.ALPHA))
    buttons.extend([b_greedy, b_alpha])
    
    # Wrap Toggle
    def toggle_wrap():
        global wrap_around
        wrap_around = not wrap_around
        setup_bot_settings()
        
    # Wrap Toggle
    bg_w = btn_img
    b_wrap = Button(window.width//2 - btn_w//2, py + 200, btn_w, 60, "", main_batch, text_group, bg_w, toggle_wrap, overlay=orbit_img, is_selected=wrap_around)
    buttons.append(b_wrap)
    
    # Start
    b_start = Button(window.width//2 - btn_w//2, py + 80, btn_w, 80, "", main_batch, text_group, btn_img, start_pve, overlay=start_img)
    buttons.append(b_start)
    
    # Back
    b_back = Button(window.width//2 - btn_w//2, py - 100, btn_w, 50, "", main_batch, text_group, btn_img, return_home, overlay=back_img)
    buttons.append(b_back)

def return_home():
    global game_state
    game_state = GameState.HOME
    reset_sprites()
    setup_home()

def show_pvp_settings():
    global game_state
    game_state = GameState.SETTINGS_POPUP
    reset_sprites()
    setup_pvp_settings()
    
def show_bot_settings():
    global game_state
    game_state = GameState.SETTINGS_POPUP
    reset_sprites()
    setup_bot_settings()

def start_pvp():
    global game_mode, game_state, players
    game_mode = GameMode.PVP
    players = ['P1', 'P2']
    game_state = GameState.PLAYING
    start_game()

def start_pve():
    global game_mode, game_state, players
    game_mode = GameMode.PVE
    players = ['P1', 'Bot']
    game_state = GameState.PLAYING
    greedy_bot.wrap_around = wrap_around
    start_game()

def reset_sprites():
    global last_move_sprite
    for s in sprites:
        s.delete()
    sprites.clear()
    
    for s in grid_sprites:
        s.delete()
    grid_sprites.clear()
    
    for l in lines:
        l.delete()
    lines.clear()
    
    for b in buttons:
        b.delete()
    buttons.clear()
    
    if last_move_sprite:
        last_move_sprite.delete()
        last_move_sprite = None
        
    for d in orbit_dots:
        d.delete()
    orbit_dots.clear()
    
    active_lines.clear()
    history_lines.clear()
    
    for f in scheduled_functions:
        pyglet.clock.unschedule(f)
    scheduled_functions.clear()

def create_orbit_dots():
    if not wrap_around: return 
    
    dot_radius = 5
    color = (255, 255, 255)
    
    # Margins are roughly gridX - space, gridY - space
    def add_dot(x, y):
        d = shapes.Circle(x, y, dot_radius, color=color, batch=main_batch, group=board_back_group)
        d.opacity = 150
        orbit_dots.append(d)
    
    # Top & Bottom
    for c in range(no_of_cells):
        cx = gridX + c * (grid_size + space) + grid_size // 2
        # Top (Above row 7)
        top_y = gridY + no_of_cells * (grid_size + space) 
        # Bottom (Below row 0)
        bot_y = gridY - space * 2
        
        # Align dots with virtual rows/cols
        # Row 8:
        y8 = gridY + 8 * (grid_size + space) + grid_size // 2
        y_neg1 = gridY + (-1) * (grid_size + space) + grid_size // 2
        
        add_dot(cx, y8)
        add_dot(cx, y_neg1)
        
    # Left & Right
    for r in range(no_of_cells):
        cy = gridY + r * (grid_size + space) + grid_size // 2
        # Right (Col 8)
        x8 = gridX + 8 * (grid_size + space) + grid_size // 2
        # Left (Col -1)
        x_neg1 = gridX + (-1) * (grid_size + space) + grid_size // 2
        
        add_dot(x8, cy)
        add_dot(x_neg1, cy)

def start_game():
    global board, scores, current_player, SOS, hc, selected_cell, last_move_pos, last_move_sprite
    reset_sprites()
    board = [[' ' for _ in range(no_of_cells)] for _ in range(no_of_cells)]
    scores = {'P1': 0, 'P2': 0}
    if game_mode == GameMode.PVE: scores = {'P1': 0, 'Bot': 0}
    
    current_player = 0
    SOS = []
    hc = None
    selected_cell = None
    last_move_pos = None
    last_move_sprite = None
    
    create_orbit_dots()
    
    # Register effects
    pyglet.clock.schedule_interval(update_effects, 1/60.0)
    scheduled_functions.append(update_effects)
    
    # Create Grid
    for i in range(no_of_cells):
        for j in range(no_of_cells):
            x = gridX + (grid_size + space) * i + space // 2
            y = gridY + (grid_size + space) * j + space // 2
            s = pyglet.sprite.Sprite(cell_bg, x=x, y=y, batch=main_batch, group=board_back_group)
            s.scale = grid_size / max(cell_bg.width, cell_bg.height)
            grid_sprites.append(s)
            
    # Back Button
    b_back = Button(20, window.height - 80, 200, 60, "", main_batch, ui_group, btn_img, return_home, overlay=back_img)
    buttons.append(b_back)


# GAME LOGIC


def check_win():
    global scores, current_player, SOS
    
    found_sos = False
    new_sos_formed = False
    
    # Archive active lines to history on new score
    flushed_history = False
    
    def is_sos(c, r, dc, dr):
        o_raw_r, o_raw_c = r + dr, c + dc
        s2_raw_r, s2_raw_c = r + 2*dr, c + 2*dc
        o_r, o_c = o_raw_r % no_of_cells, o_raw_c % no_of_cells
        s2_r, s2_c = s2_raw_r % no_of_cells, s2_raw_c % no_of_cells
        
        if not wrap_around:
            if not (0 <= o_raw_r < no_of_cells and 0 <= o_raw_c < no_of_cells): return None, None, False
            if not (0 <= s2_raw_r < no_of_cells and 0 <= s2_raw_c < no_of_cells): return None, None, False
        
        if board[r][c] == 'S' and board[o_r][o_c] == 'O' and board[s2_r][s2_c] == 'S':
            raw = ((r, c), (o_r, o_c), (s2_r, s2_c))
            sorted_v = tuple(sorted(raw))
            wrapped = (o_raw_r != o_r or o_raw_c != o_c) or (s2_raw_r != s2_r or s2_raw_c != s2_c)
            return raw, sorted_v, wrapped
        return None, None, False

    for r in range(no_of_cells):
        for c in range(no_of_cells):
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                raw, sorted_v, wrapped = is_sos(c, r, dc, dr)
                if raw and sorted_v not in SOS:
                    if not flushed_history:
                        # Archive active lines to history
                        if active_lines:
                             history_lines.append(list(active_lines))
                             active_lines.clear()
                        
                        recalc_history_opacity()
                        flushed_history = True
                        
                    SOS.append(sorted_v)
                    scores[players[current_player]] += 1
                    found_sos = True
                    draw_win_line(raw, current_player, dotted=wrapped)

    return found_sos

def draw_win_line(raw_coords, player_idx, dotted=False):
    s1, o, s2 = raw_coords
    
    def to_screen_center(r, c):
        x = gridX + c * (grid_size + space) + grid_size // 2
        y = gridY + r * (grid_size + space) + grid_size // 2
        return x, y

    def is_wrapped(r1, c1, r2, c2):
        return abs(r1 - r2) > 1 or abs(c1 - c2) > 1

    def draw_segment(r1, c1, r2, c2):
        if is_wrapped(r1, c1, r2, c2):
             vr, vc = r2, c2
             if abs(r1 - r2) > 1: vr = r1 + 1 if r2 < r1 else r1 - 1 
             if abs(c1 - c2) > 1: vc = c1 + 1 if c2 < c1 else c1 - 1
             
             x1, y1 = to_screen_center(r1, c1)
             x2, y2 = to_screen_center(vr, vc) # Dot position
             
             # Shorten line by 10% to create gap
             x2_new = x1 + (x2 - x1) * 0.9
             y2_new = y1 + (y2 - y1) * 0.9
             
             draw_dashed_line(x1, y1, x2_new, y2_new, player_idx, segments=8) 
             
             vr2, vc2 = r1, c1
             if abs(r1 - r2) > 1: vr2 = r2 - 1 if r1 > r2 else r2 + 1 
             if abs(c1 - c2) > 1: vc2 = c2 - 1 if c1 > c2 else c2 + 1
             
             x3, y3 = to_screen_center(vr2, vc2) # Dot position
             x4, y4 = to_screen_center(r2, c2)   # Cell position
             
             # Shorten line by 10% to create gap
             x3_new = x3 + (x4 - x3) * 0.1
             y3_new = y3 + (y4 - y3) * 0.1
             
             draw_dashed_line(x3_new, y3_new, x4, y4, player_idx, segments=8)
             
        else:
             x1, y1 = to_screen_center(r1, c1)
             x2, y2 = to_screen_center(r2, c2)
             create_line(x1, y1, x2, y2, player_idx)

    # Process S1 -> O
    draw_segment(s1[0], s1[1], o[0], o[1])
    # Process O -> S2
    draw_segment(o[0], o[1], s2[0], s2[1])

def draw_dashed_line(x1, y1, x2, y2, p_idx, segments=6):
    colors = [(0, 183, 239), (237, 28, 36)]
    total_len = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    if total_len == 0: return
    
    dx = (x2 - x1) / total_len
    dy = (y2 - y1) / total_len
    
    # Dense dots: 1 dot every 10px
    num_dots = int(total_len / 10) 
    if num_dots < 3: num_dots = 3
    
    step = total_len / num_dots
    dash_len = step * 0.6
    
    for i in range(num_dots):
        dist = i * step
        
        sx = x1 + dx * dist
        sy = y1 + dy * dist
        
        ex = sx + dx * dash_len
        ey = sy + dy * dash_len
        
        l = shapes.Line(sx, sy, ex, ey, thickness=line_thickness, color=colors[p_idx], batch=main_batch, group=ui_group)
        l.opacity = 255 # Start Bright
        lines.append(l)
        active_lines.append(l)

def create_line(x1, y1, x2, y2, p_idx, is_dot=False):
    colors = [(0, 183, 239), (237, 28, 36)]
    l = shapes.Line(x1, y1, x2, y2, thickness=line_thickness, color=colors[p_idx], batch=main_batch, group=ui_group)
    l.opacity = 255 # Start bright
    lines.append(l)
    active_lines.append(l)

def place_symbol(r, c, symbol):
    global last_move_pos, last_move_sprite
    board[r][c] = symbol
    
    # 1. Update Last Move
    last_move_pos = (r, c)
    if last_move_sprite: last_move_sprite.delete()
    
    lm_x = gridX + c * (grid_size + space) + space // 2
    lm_y = gridY + r * (grid_size + space) + space // 2
    last_move_sprite = pyglet.sprite.Sprite(last_move_img, x=lm_x, y=lm_y, batch=main_batch, group=highlight_group)
    last_move_sprite.scale = grid_size / max(last_move_img.width, last_move_img.height)
    
    # 2. Place Symbol
    img = img_s if symbol == 'S' else img_o
    x = gridX + c * (grid_size + space) + 0.1 * grid_size
    y = gridY + r * (grid_size + space) + 0.1 * grid_size
    s = pyglet.sprite.Sprite(img, x=x, y=y, batch=main_batch, group=board_main_group)
    s.scale = (0.8 * grid_size) / max(img.width, img.height)
    sprites.append(s)

def end_game():
    global game_state, winner_text
    game_state = GameState.GAME_OVER
    s1 = scores[players[0]]
    s2 = scores[players[1]]
    if s1 > s2: winner_text = f"{players[0]} WINS!"
    elif s2 > s1: winner_text = f"{players[1]} WINS!"
    else: winner_text = "DRAW!"

# INPUT & BOT CONTROL


def bot_turn_trigger(dt):
    bot_turn_execute()

def bot_turn_execute():
    global current_player, bot_moves_queue
    
    move = None
    letter = 'S'
    
    if bot_type == BotType.ALPHA:
        res = alpha_bot.choose_move_wrapper(board, SOS, scores, 1)
        if res: move, letter = res
        else: move, letter = greedy_bot.choose_move(board) 
    else:
        move, letter = greedy_bot.choose_move(board)
        
    r, c = move
    if board[r][c] != ' ': 
        print("Bot tried invalid move!")
        return 
        
    place_symbol(r, c, letter)
    scored = check_win()
    
    if scored and not is_board_full():
        pyglet.clock.schedule_once(bot_turn_trigger, 0.8)
    else:
        if is_board_full():
            end_game()
        elif not scored:
            current_player = 0 

def is_board_full():
    for r in board:
        if ' ' in r: return False
    return True

@window.event
def on_mouse_press(x, y, button, modifiers):
    for b in buttons:
        if b.on_mouse_press(x, y, button, modifiers):
            return 
            
    if game_state == GameState.PLAYING and ((game_mode == GameMode.PVE and current_player == 0) or game_mode == GameMode.PVP):
        if gridX <= x <= gridX + cellSize and gridY <= y <= gridY + cellSize:
            c = int((x - gridX) // (grid_size + space))
            r = int((y - gridY) // (grid_size + space))
            
            if 0 <= r < no_of_cells and 0 <= c < no_of_cells:
                if board[r][c] == ' ':
                    global selected_cell, hc
                    selected_cell = (r, c)
                    if hc: hc.delete()
                    begX = gridX + c * (grid_size + space)
                    begY = gridY + r * (grid_size + space)
                    hc = pyglet.sprite.Sprite(cell_sel, x=begX, y=begY, batch=main_batch, group=board_back_group)
                    hc.scale = grid_size / max(cell_sel.width, cell_sel.height)

@window.event
def on_mouse_release(x, y, button, modifiers):
    for b in buttons:
        b.on_mouse_release(x, y, button, modifiers)

@window.event
def on_mouse_motion(x, y, dx, dy):
    for b in buttons:
        b.check_hover(x, y)

@window.event
def on_key_press(symbol, modifiers):
    global current_player, hc, selected_cell, game_state
    
    if game_state == GameState.GAME_OVER:
        if symbol == key.ENTER or symbol == key.SPACE:
            return_home()
        return

    if game_state == GameState.PLAYING and selected_cell:
        if (game_mode == GameMode.PVE and current_player == 1): return 
        
        r, c = selected_cell
        char = None
        if symbol == key.S: char = 'S'
        elif symbol == key.O: char = 'O'
        
        if char and board[r][c] == ' ':
            place_symbol(r, c, char)
            if hc: 
                 hc.delete()
                 hc = None
            selected_cell = None
            scored = check_win()
            if is_board_full():
                end_game()
            else:
                if not scored:
                    current_player = 1 - current_player
                    if game_mode == GameMode.PVE and current_player == 1:
                        pyglet.clock.schedule_once(bot_turn_trigger, 0.5)

def draw_outline_label(text, x, y, font_name, font_size, color, anchor_x='center', stroke_width=2):
    # Draw black outline
    offsets = []
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if dx == 0 and dy == 0: continue
            if dx**2 + dy**2 > stroke_width**2 + 1: continue 
            offsets.append((dx, dy))
            
    for dx, dy in offsets:
        pyglet.text.Label(text, font_name=font_name, font_size=font_size, color=(0, 0, 0, 255),
                          x=x+dx, y=y+dy, anchor_x=anchor_x).draw()
                          
    # Draw main text
    pyglet.text.Label(text, font_name=font_name, font_size=font_size, color=color,
                      x=x, y=y, anchor_x=anchor_x).draw()

@window.event
def on_draw():
    window.clear()
    bg_img.blit(0, 0, width=window.width, height=window.height)
    main_batch.draw()
    
    if game_state == GameState.PLAYING or game_state == GameState.GAME_OVER:
        p1_score = scores[players[0]]
        p2_score = scores[players[1]]
        top_y = window.height - 50
        
        # Calculate available margin
        grid_pix_w = no_of_cells * (grid_size + space)
        margin = (window.width - grid_pix_w) / 2
        
        # Center of Left Margin
        x_p1 = margin / 2
        # Center of Right Margin
        x_p2 = window.width - (margin / 2)
        
        # Draw Outlined Scores
        draw_outline_label(f"{players[0]}: {p1_score}", x_p1, window.height // 2, 
                           custom_font, 30, (0, 183, 239, 255), stroke_width=2)
                           
        draw_outline_label(f"{players[1]}: {p2_score}", x_p2, window.height // 2, 
                           custom_font, 30, (237, 28, 36, 255), stroke_width=2)
                          
        if game_state == GameState.GAME_OVER:
             pyglet.text.Label(f"GAME OVER: {winner_text}", font_name=custom_font, font_size=32, 
                              x=window.width//2, y=window.height//2, anchor_x='center', anchor_y='center', color=(255, 255, 0, 255)).draw()
             pyglet.text.Label("Press ENTER to menu", font_name=custom_font, font_size=18, 
                              x=window.width//2, y=window.height//2 - 50, anchor_x='center', anchor_y='center').draw()
        else:
             turn_str = f"{players[current_player]}'s Turn"
             if game_mode == GameMode.PVE and current_player == 1: turn_str = "Bot Thinking..."
             pyglet.text.Label(turn_str, font_name=custom_font, font_size=24,
                               x=window.width//2, y=top_y, anchor_x='center').draw()

# Init
pyglet.clock.schedule_interval(update_ui, 1/60.0)
setup_home()
pyglet.app.run()
