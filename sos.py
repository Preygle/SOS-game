import pyglet
from pyglet import shapes, font
from pyglet import shapes, font
from pyglet.window import key
from enum import Enum
import random
import math
# import torch
import numpy as np
import os
import sys

# Import Bot Logic
from greedy_bot import SOSBot

# Helper: Blend Images using Numpy
def blend_images(bg_img, overlay_img):
    if not overlay_img: return bg_img
    
    # Get Numpy arrays
    def get_arr(img):
        if not hasattr(img, 'get_image_data'): img = img.get_texture()
        idata = img.get_image_data()
        data = idata.get_data('RGBA', idata.width * 4)
        return np.frombuffer(data, dtype=np.uint8).reshape((idata.height, idata.width, 4))
    
    bg_arr = get_arr(bg_img)
    ov_arr = get_arr(overlay_img)
    
    # Resize overlay to match BG (Nearest Neighbor)
    h, w = bg_arr.shape[:2]
    oh, ow = ov_arr.shape[:2]
    
    if (oh, ow) != (h, w):
        row_idx = (np.arange(h) * (oh / h)).astype(int)
        col_idx = (np.arange(w) * (ow / w)).astype(int)
        ov_arr = ov_arr[row_idx[:, None], col_idx]
        
    # Alpha Blend
    # Normalize to 0-1
    src = ov_arr.astype(float) / 255.0
    dst = bg_arr.astype(float) / 255.0
    
    # Simplified Alpha Compositing
    src_rgb = src[..., :3]
    src_a = src[..., 3:4] # (H,W,1)
    
    dst_rgb = dst[..., :3]
    dst_a = dst[..., 3:4]
    
    # Output Alpha
    out_a = src_a + dst_a * (1.0 - src_a)
    
    # Output RGB
    # Prevent division by zero
    safe_alpha = np.maximum(out_a, 1e-6)
    out_rgb = (src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)) / safe_alpha
    
    out = np.dstack((out_rgb, out_a)) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    # Convert back to ImageData
    return pyglet.image.ImageData(w, h, 'RGBA', out.tobytes())

# Helper: Blend Images using Numpy
def blend_images(bg_img, overlay_img):
    if not overlay_img: return bg_img
    
    # Get Numpy arrays
    def get_arr(img):
        if not hasattr(img, 'get_image_data'): img = img.get_texture()
        idata = img.get_image_data()
        data = idata.get_data('RGBA', idata.width * 4)
        return np.frombuffer(data, dtype=np.uint8).reshape((idata.height, idata.width, 4))
    
    bg_arr = get_arr(bg_img)
    ov_arr = get_arr(overlay_img)
    
    # Resize overlay to match BG (Nearest Neighbor)
    h, w = bg_arr.shape[:2]
    oh, ow = ov_arr.shape[:2]
    
    if (oh, ow) != (h, w):
        row_idx = (np.arange(h) * (oh / h)).astype(int)
        col_idx = (np.arange(w) * (ow / w)).astype(int)
        ov_arr = ov_arr[row_idx[:, None], col_idx]
        
    # Alpha Blend
    # Normalize to 0-1
    src = ov_arr.astype(float) / 255.0
    dst = bg_arr.astype(float) / 255.0
    
    # Simplified Alpha Compositing
    src_rgb = src[..., :3]
    src_a = src[..., 3:4] # (H,W,1)
    
    dst_rgb = dst[..., :3]
    dst_a = dst[..., 3:4]
    
    # Output Alpha
    out_a = src_a + dst_a * (1.0 - src_a)
    
    # Output RGB
    # Prevent division by zero
    safe_alpha = np.maximum(out_a, 1e-6)
    out_rgb = (src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)) / safe_alpha
    
    out = np.dstack((out_rgb, out_a)) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    # Convert back to ImageData
    return pyglet.image.ImageData(w, h, 'RGBA', out.tobytes())


# BOT WRAPPER (From sos_bot.py)

class AlphaBot:
    def __init__(self, model_path='checkpoints/best.pth'):
        print("AlphaBot is currently a placeholder.")

    def choose_move_wrapper(self, board, current_sos, current_scores, current_player_idx):
        return None


# UI UTILS & BUTTON CLASS


# Helper: Blend Images using Numpy
def blend_images(bg_img, overlay_img):
    if not overlay_img: return bg_img
    
    # Get Numpy arrays
    def get_arr(img):
        if not hasattr(img, 'get_image_data'): img = img.get_texture()
        idata = img.get_image_data()
        data = idata.get_data('RGBA', idata.width * 4)
        return np.frombuffer(data, dtype=np.uint8).reshape((idata.height, idata.width, 4))
    
    bg_arr = get_arr(bg_img)
    ov_arr = get_arr(overlay_img)
    
    # Resize overlay to match BG (Nearest Neighbor)
    h, w = bg_arr.shape[:2]
    oh, ow = ov_arr.shape[:2]
    
    if (oh, ow) != (h, w):
        row_idx = (np.arange(h) * (oh / h)).astype(int)
        col_idx = (np.arange(w) * (ow / w)).astype(int)
        ov_arr = ov_arr[row_idx[:, None], col_idx]
        
    # Alpha Blend
    # Normalize to 0-1
    src = ov_arr.astype(float) / 255.0
    dst = bg_arr.astype(float) / 255.0
    
    # Simplified Alpha Compositing
    src_rgb = src[..., :3]
    src_a = src[..., 3:4] # (H,W,1)
    
    dst_rgb = dst[..., :3]
    dst_a = dst[..., 3:4]
    
    # Output Alpha
    out_a = src_a + dst_a * (1.0 - src_a)
    
    # Output RGB
    # Prevent division by zero
    safe_alpha = np.maximum(out_a, 1e-6)
    out_rgb = (src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)) / safe_alpha
    
    out = np.dstack((out_rgb, out_a)) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    # Convert back to ImageData
    return pyglet.image.ImageData(w, h, 'RGBA', out.tobytes())

class Button:
    def __init__(self, x, y, width, height, text, batch, group, img_normal, callback=None, font_size=18, overlay=None, is_selected=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.callback = callback
        self.is_selected = is_selected
        self.batch = batch
        self.orig_group = group
        
        # Merge Images: "Merge it actually in a new image"
        self.img_normal = img_normal
        self.img_merged = blend_images(img_normal, overlay) if overlay else img_normal
        
        # Scale based on merged image size
        self.base_scale_x = width / self.img_merged.width
        self.base_scale_y = height / self.img_merged.height
        
        # Sprite uses pure unified texture
        self.sprite = pyglet.sprite.Sprite(self.img_merged, x=x, y=y, batch=batch, group=group)
        self.sprite.scale_x = self.base_scale_x
        self.sprite.scale_y = self.base_scale_y
        
        # Force much higher Z-order for text to ensure it sits above all strips
        self.text_group_parent = animation_text_group 
        
        self.strips = [] # For CPU Animation
        
        self.label = None
        if text:
            self.label = pyglet.text.Label(text, font_name=custom_font, font_size=font_size,
                                           x=int(x + width//2), y=int(y + height//2),
                                           anchor_x='center', anchor_y='center',
                                           color=(255, 255, 255, 255),
                                           batch=batch, group=self.text_group_parent)
        self.total_time = 0.0

    def create_strips(self):
        if self.strips: return
        
        # Slice the MERGED image
        img = self.img_merged
        if not hasattr(img, 'get_region'): img = img.get_texture()
        total_w = img.width
        chunk_w = max(1, total_w // SINE_CHUNKS)
        
        for i in range(SINE_CHUNKS):
            x = i * chunk_w
            if x >= total_w: break
            w = min(chunk_w, total_w - x)
            # Create strip region from the combined texture
            region = img.get_region(x, 0, w, img.height)
            s = pyglet.sprite.Sprite(region, batch=self.batch, group=self.orig_group)
            s.anim_x_offset = x 
            self.strips.append(s)

    def delete_strips(self):
        for s in self.strips:
            s.delete()
        self.strips.clear()

    def update(self, dt):
        self.total_time += dt
        
        hover_anim = 1.1 if hasattr(self, 'is_hovered') and self.is_hovered else 1.0
        
        final_scale = 1.05 if self.is_selected else 1.0
        if hasattr(self, 'is_hovered') and self.is_hovered:
             final_scale += 0.05
        
        final_sx = self.base_scale_x * final_scale
        final_sy = self.base_scale_y * final_scale
        
        current_w = self.img_merged.width * final_sx
        current_h = self.img_merged.height * final_sy
        
        x_shift = (current_w - self.width) / 2
        y_shift = (current_h - self.height) / 2
        
        base_x = self.x - x_shift
        base_y = self.y - y_shift
        
        # Apply to Main Sprite (Static)
        self.sprite.scale_x = final_sx
        self.sprite.scale_y = final_sy
        self.sprite.x = base_x
        self.sprite.y = base_y
        
        # Strip Animation (Only if selected)
        if self.is_selected:
            if not self.strips:
                self.create_strips()
                self.sprite.visible = False # Hide main sprite
                
            # Update Strips
            for s in self.strips:
                s.scale_x = final_sx
                s.scale_y = final_sy
                s.x = base_x + s.anim_x_offset * final_sx
                
                phase_x = s.x - base_x
                
                angle = SINE_FREQ * 0.01 * phase_x + self.total_time * SINE_SPEED
                y_off = SINE_AMP * math.sin(angle)
                s.y = base_y + y_off
            
            # Text (Label)
            if self.label:
                 center_angle = SINE_FREQ * 0.01 * (current_w/2) + self.total_time * SINE_SPEED
                 center_y_off = SINE_AMP * math.sin(center_angle)
                 self.label.y = int((self.y + self.height//2) + center_y_off)
                 self.label.x = int(self.x + self.width//2)
                 
        else:
            if self.strips:
                self.delete_strips()
                self.sprite.visible = True
                
            if self.label: 
                self.label.y = self.y + self.height//2
                self.label.x = self.x + self.width//2
        
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
        if self.label: self.label.delete()
        self.delete_strips()


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
history_img = pyglet.resource.image('History.png')

# Batches
main_batch = pyglet.graphics.Batch()
bg_group = pyglet.graphics.Group(order=0)
board_back_group = pyglet.graphics.Group(order=1)
highlight_group = pyglet.graphics.Group(order=2)
board_main_group = pyglet.graphics.Group(order=3) 
ui_group = pyglet.graphics.Group(order=4)         
text_group = pyglet.graphics.Group(order=6)
# High Z-order group for animated text to ensure visibility
history_panel_group = pyglet.graphics.Group(order=10)
hud_group = pyglet.graphics.Group(order=12)
# High Z-order group for animated text to ensure visibility
animation_text_group = pyglet.graphics.Group(order=20)

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
active_lines = []
history_lines = []
grid_labels = {}
match_history = []
show_history = False
history_bg_shape = None
history_border_shape = None
hist_btn_bg = None
hist_btn_tri = None

# Game State Globals (Initialized here to avoid NameError in reset_sprites)
hc = None
selected_cell = None
last_move_pos = None
last_move_sprite = None
board = []
scores = {}
current_player = 0
SOS = []
bot_turn_trigger = None # Placeholder if it's a variable, or function name? 
# If bot_turn_trigger is a function, we shouldn't overwrite it with None if defined later.
# But python functions are hoisted if defined with `def`.
# If `bot_turn_trigger` is `def` defined later, `unschedule` works with the name.

# Actually, if `bot_turn_trigger` is `def`, I don't need to init it.
# If `hc` is variable, I do.


# ==========================================
# 5. SHADERS (SINE WAVE)
# ==========================================

# Animation Parameters (Editable)
SINE_AMP = 16
SINE_FREQ = 2.5
SINE_SPEED = 10
SINE_CHUNKS = 32

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
    global last_move_sprite, hc, selected_cell, history_bg_shape, show_history
    global hist_btn_bg, hist_btn_tri
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

    for k, l in grid_labels.items():
        l.delete()
    grid_labels.clear()

    match_history.clear()
    
    # Remove history panel shape from batch visible logic
    if history_bg_shape: 
         history_bg_shape.visible = False
    
    # Hide any lingering selected cell highlight
    if hc:
        hc.delete()
        hc = None
        
    selected_cell = None
    show_history = False
    
    for f in scheduled_functions:
        pyglet.clock.unschedule(f)
    scheduled_functions.clear()
    
    # Unschedule any potential one-off bot triggers
    pyglet.clock.unschedule(bot_turn_trigger)

    if hist_btn_bg: hist_btn_bg.delete(); hist_btn_bg = None
    if hist_btn_tri: hist_btn_tri.delete(); hist_btn_tri = None

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
    global board, scores, current_player, SOS, hc, selected_cell, last_move_pos, last_move_sprite, show_history
    global history_bg_shape, history_border_shape
    global hist_btn_bg, hist_btn_tri
    
    reset_sprites()
    board = [[' ' for _ in range(no_of_cells)] for _ in range(no_of_cells)]
    scores = {'P1': 0, 'P2': 0}
    show_history = False
    
    # Create History Panel Shapes (Hidden by default)
    # The user wants "reduction from both top and bottom such that the history area dosent encompasses the history button"
    # Button is at (width-60, height-60)
    # So we should start the panel below the button
    
    panel_w = 350 
    panel_x = window.width - panel_w
    
    # Let's say button area is top 80 pixels
    # And bottom area is 50 pixels
    # Panel Y = 50
    # Panel Height = window.height - 80 - 50 = window.height - 130
    
    # Create History Panel Shapes (Hidden by default)
    # User requested: Rounded Rect, Color #33203F (51, 32, 63), Alpha 0.7 (179)
    # Centered in right margin
    
    panel_w = 350
    
    # Calculate Right Margin Center
    # Grid End X = gridX + total_grid_dim
    # But wait, gridX is calculated at module level.
    # gridX = window.width // 2 - total_grid_dim // 2
    # So Right Margin moves from (gridX + total_grid_dim) to window.width
    
    grid_end_x = gridX + total_grid_dim
    margin_width = window.width - grid_end_x
    
    # Center of margin
    margin_center_x = grid_end_x + margin_width // 2
    panel_x = margin_center_x - panel_w // 2
    
    panel_y = 50
    panel_h = window.height - 130
    
    # Try using RoundedRectangle. If fails (old pyglet), fallback to Rectangle
    try:
        history_bg_shape = shapes.RoundedRectangle(panel_x, panel_y, panel_w, panel_h, radius=20, color=(51, 32, 63, 179), batch=main_batch, group=history_panel_group)
    except AttributeError:
        # Fallback for older pyglet
        history_bg_shape = shapes.Rectangle(panel_x, panel_y, panel_w, panel_h, color=(51, 32, 63, 179), batch=main_batch, group=history_panel_group)
        
    history_bg_shape.visible = False

    # Remove border shape as it conflicts with rounded look (or would need to be rounded too)
    history_border_shape = None
    
    # Initialize scores for ALL players to avoid KeyError
    scores = {p: 0 for p in players}
    
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

            # Add Chess Coordinates (A1..H8) on Bottom-Right
            # Cols: A -> H (Left -> Right)
            # Rows: 1 -> 8 (Top -> Bottom)
            # Rows: 1 -> 8 (Top -> Bottom) -> Now User requested Bottom=1
            col_str = chr(ord('A') + i)
            row_str = str(j + 1)
            label_text = f"{col_str}{row_str}"
            
            # Position: Bottom Right of the cell
            # Cell BG Anchor is (0,0) based on logic
            # Label Anchor: Right, Bottom
            lbl_x = x + grid_size * 0.90
            lbl_y = y + grid_size * 0.10
            
            f_size = int(grid_size * 0.15)
            if f_size < 8: f_size = 8
            
            l = pyglet.text.Label(label_text, font_name=custom_font, font_size=f_size,
                                  x=lbl_x, y=lbl_y, anchor_x='right', anchor_y='bottom',
                                  color=(233, 30, 99, 130), batch=main_batch, group=highlight_group)
            grid_labels[(j, i)] = l
            
    # Back Button
    b_back = Button(20, window.height - 80, 200, 60, "", main_batch, ui_group, btn_img, return_home, overlay=back_img)
    buttons.append(b_back)

    # History Button (Right)
    # Using a simple "H" label on a small button for now, or repurposing btn_img
    def toggle_history():
        global show_history
        show_history = not show_history
        if history_bg_shape: history_bg_shape.visible = show_history
        # history_border_shape is removed/None
        
    # Button needs to be accessbile -> Draw ON TOP of panel (Order 12 vs 10)
    # History Toggle (Triangle Button)
    # Replaces sprite button.
    # Dimensions: 40x40. Position: Bottom-Right corner margin.
    
    tb_w = 40
    tb_h = 40
    tb_x = window.width - tb_w - 20
    tb_y = 20 # Bottom margin
    
    # Background: Rectangle (Dark Grey/Black?)
    # "red/blue" was for separator. User didn't specify button color, assuming standard UI or transparent?
    # "small rectangle having a triangle in it"
    # Let's align with the panel style: Rounded Rect or Rect.
    # Let's use Rect. Color: #33203F (Panel Color) or lighter?
    # Let's use a visible color.
    
    hist_btn_bg = shapes.Rectangle(tb_x, tb_y, tb_w, tb_h, color=(51, 32, 63), batch=main_batch, group=hud_group)
    hist_btn_bg.opacity = 200
    
    # Triangle
    # Center:
    tx = tb_x + tb_w // 2
    ty = tb_y + tb_h // 2
    ts = 10 # Size
    
    # Default: Hidden -> Face LEFT (<)
    # Tip at tx - ts, Base at tx + ts
    
    # Vertices: (TipX, TipY), (TopRightX, TopRightY), (BotRightX, BotRightY)
    # Left: (tx - ts, ty), (tx + ts, ty + ts), (tx + ts, ty - ts)
    
    hist_btn_tri = shapes.Triangle(tx - ts, ty, tx + ts, ty + ts, tx + ts, ty - ts, color=(255, 255, 255), batch=main_batch, group=hud_group)
    
    def toggle_history():
        global show_history
        show_history = not show_history
        if history_bg_shape: history_bg_shape.visible = show_history
        
        # Update Triangle
        # Shown -> Face RIGHT (>)
        # Hidden -> Face LEFT (<)
        
        if show_history:
             # Right: (tx + ts, ty), (tx - ts, ty + ts), (tx - ts, ty - ts)
             hist_btn_tri.x = tx + ts
             hist_btn_tri.y = ty
             hist_btn_tri.x2 = tx - ts
             hist_btn_tri.y2 = ty + ts
             hist_btn_tri.x3 = tx - ts
             hist_btn_tri.y3 = ty - ts
        else:
             # Left
             hist_btn_tri.x = tx - ts
             hist_btn_tri.y = ty
             hist_btn_tri.x2 = tx + ts
             hist_btn_tri.y2 = ty + ts
             hist_btn_tri.x3 = tx + ts
             hist_btn_tri.y3 = ty - ts
             
    # Bind click event manually in on_mouse_press since we aren't using Button class
    # We'll need a way to reference this callback.
    # We can assign it to `hist_btn_bg.callback` and check it in on_mouse_press.
    hist_btn_bg.callback = toggle_history


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
                    
                    # Record History
                    # (current_player, raw_coords) -> match_history
                    # Format: "A1, B2, C3"
                    # Highlight the last move in YELLOW (#FFFF00)
                    
                    def coords_to_str(r, c):
                        col = chr(ord('A') + c)
                        row = str(r + 1)
                        return f"{col}{row}"
                        
                    s1, o, s2 = raw
                    
                    # Determine colors
                    # P1 (Blue): #00B7EF
                    # P2 (Red): #ED1C24
                    # Highlight: #FFFF00
                    
                    base_color = "#00B7EF" if current_player == 0 else "#ED1C24"
                    hl_color = "#FFFF00"
                    
                    parts = []
                    for coord in [s1, o, s2]:
                        txt = coords_to_str(*coord)
                        if coord == last_move_pos:
                            parts.append(f"<font color='{hl_color}'>{txt}</font>")
                        else:
                            parts.append(f"<font color='{base_color}'>{txt}</font>")
                            
                    separator = f"<font color='{base_color}'> ✧ </font>"
                    notation = separator.join(parts)
                    # Wrap in a font tag to ensure size/font is applied if needed, but HTMLLabel handles it.
                    # We store the HTML string directly.
                    match_history.append({'player': current_player, 'notation': notation})

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
    
    # Hide coordinate label
    if (r, c) in grid_labels:
        grid_labels[(r, c)].visible = False
    
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
            
    # Check History Toggle Button
    # It's a global shape `hist_btn_bg`
    if game_state == GameState.PLAYING or game_state == GameState.GAME_OVER:
        if hist_btn_bg:
             # AABB Check
             if hist_btn_bg.x <= x <= hist_btn_bg.x + hist_btn_bg.width and \
                hist_btn_bg.y <= y <= hist_btn_bg.y + hist_btn_bg.height:
                    if hasattr(hist_btn_bg, 'callback'):
                        hist_btn_bg.callback()
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
        
        if show_history:
             # Reposition Header & Content
             # Use the actual shape position to ensure alignment
             if history_bg_shape:
                 panel_x = history_bg_shape.x
                 panel_w = history_bg_shape.width
             else:
                 # Fallback if shape missing (shouldn't happen)
                 panel_w = 350
                 panel_x = window.width - panel_w
             
             # Header
             # Inside the new panel area (Top)
             panel_top = 50 + (window.height - 130)
             
             # Color F2E9FF -> (242, 233, 255)
             pyglet.text.Label("HISTORY", font_name=custom_font, font_size=16,
                               x=panel_x + panel_w//2, y=panel_top - 30, anchor_x='center',
                               color=(242, 233, 255, 255)).draw()
                               
             # List Items
             start_y = panel_top - 60
             for i, item in enumerate(reversed(match_history)):
                  y_pos = start_y - i * 30
                  if y_pos < 50 + 20: break # Stop before bottom margin
                  
                  p_idx = item['player']
                  txt = item['notation']
                  # txt is now HTML string with colors embedded
                  
                  # Use HTMLLabel
                  # Note: HTMLLabel anchors might behave differently.
                  # It supports 'center', 'left', 'right'.
                  # We want left aligned in panel? Or centered?
                  # Center valid sequence with history
                  centered_txt = f"<center>{txt}</center>"
                  
                  # Use HTMLLabel with center alignment
                  lbl = pyglet.text.HTMLLabel(centered_txt, x=panel_x + panel_w // 2, y=y_pos, width=panel_w-20, multiline=True, anchor_x='center', anchor_y='center', batch=None)
                  lbl.font_name = custom_font
                  lbl.font_size = 12
                  lbl.draw()

             # Scores on Left (Stacked)
             # "1/3 distance from the screen height (equal distance)"
             # P1 at 2/3 Height, P2 at 1/3 Height
             # This creates 3 equal vertical segments (Top->P1, P1->P2, P2->Bot)
             x_scores = margin / 2
             y_p1 = window.height * (2/3)
             y_p2 = window.height * (1/3)
             
             draw_outline_label(f"{players[0]}: {p1_score}", x_scores, y_p1, 
                                custom_font, 30, (0, 183, 239, 255), stroke_width=2)
                                
             draw_outline_label(f"{players[1]}: {p2_score}", x_scores, y_p2, 
                                custom_font, 30, (237, 28, 36, 255), stroke_width=2)
        else:
            # Normal Layout
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
