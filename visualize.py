import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import pyglet
from pyglet import shapes
import os
import sys

# Import game logic to drive the replay
from game_logic import SOSGame  
# Note: original game_logic.py has SOSGame. 
# We need to replicate the visual logic from sos_1v1.py but automated.

# ---------------- PLOTTING ----------------
def plot_stats(log_file='training_log.csv'):
    if not os.path.exists(log_file):
        print(f"No log file found at {log_file}")
        return
        
    df = pd.read_csv(log_file)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left Axis: Policy Loss
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Policy Loss', color=color)
    ax1.plot(df['Iteration'], df['PolicyLoss'], color=color, label='Policy Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Right Axis: Value Loss
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Value Loss (MSE)', color=color)
    ax2.plot(df['Iteration'], df['ValueLoss'], color=color, linestyle='--', label='Value Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.0) # Normalize Value Loss to 0-1 range
    
    plt.title('AlphaZero Training Progress')
    fig.tight_layout()
    plt.show()

# ---------------- REPLAY VIEWER ----------------
class ReplayWindow(pyglet.window.Window):
    def __init__(self, replay_file):
        super().__init__(800, 800, caption=f"Replay: {replay_file}")
        
        with open(replay_file, 'r') as f:
            self.data = json.load(f)
            
        self.moves = self.data['moves']
        self.move_idx = 0
        self.board_size = 8
        self.game = SOSGame()
        
        self.grid_cells = [[' ' for _ in range(8)] for _ in range(8)]
        self.sprites = []
        self.lines = []
        self.batch = pyglet.graphics.Batch()
        
        # Resources
        pyglet.resource.path = ['assets']
        pyglet.resource.reindex()
        self.img_s = pyglet.resource.image('s.png')
        self.img_o = pyglet.resource.image('o.png')
        self.img_bg = pyglet.resource.image('background.png')
        self.img_cell = pyglet.resource.image('cell.png')
        
        # Grid Dimensions
        self.margin = 50
        self.grid_size = 800 - 2 * self.margin
        self.cw = self.grid_size / 8
        
        # Create Grid Lines (Static)
        self.shapes = []
        # Vertical lines
        for i in range(9):
            x = self.margin + i * self.cw
            line = shapes.Line(x, self.margin, x, 800 - self.margin, thickness=3, color=(200, 200, 200), batch=self.batch)
            self.shapes.append(line)
        # Horizontal lines
        for i in range(9):
            y = self.margin + i * self.cw
            line = shapes.Line(self.margin, y, 800 - self.margin, y, thickness=3, color=(200, 200, 200), batch=self.batch)
            self.shapes.append(line)
            
        # Draw cell backgrounds
        for r in range(8):
            for c in range(8):
                x = self.margin + c * self.cw + self.cw/2
                # Invert Y for Rows: Row 0 is at TOP (High Y)
                # Grid bottom is at self.margin. Grid top is at 800-margin.
                # Row 0 y-center = (800 - margin) - cw/2
                # Row r y-center = (800 - margin) - r*cw - cw/2
                y = (800 - self.margin) - r * self.cw - self.cw/2
                
                # Cell Sprite
                bg = pyglet.sprite.Sprite(self.img_cell, x=x, y=y, batch=self.batch)
                bg.image.anchor_x = bg.width // 2
                bg.image.anchor_y = bg.height // 2
                bg.scale = (self.cw * 0.95) / max(bg.width, bg.height)
                self.sprites.append(bg) # Keep ref
        
        # Schedule update
        pyglet.clock.schedule_interval(self.update_game, 0.5)

    def update_game(self, dt):
        if self.move_idx < len(self.moves):
            action = self.moves[self.move_idx]
            
            # Sync with Game Logic
            _, _, _, _ = self.game.step(action)
            
            # Decode action (0-63=S, 64-127=O)
            is_o = action >= 64
            idx = action - 64 if is_o else action
            r = idx // 8
            c = idx % 8
            
            # Graphic: Place Piece
            # Center of cell (Row 0 at Top)
            x_center = self.margin + c * self.cw + self.cw/2
            y_center = (800 - self.margin) - r * self.cw - self.cw/2
            
            img = self.img_o if is_o else self.img_s
            s = pyglet.sprite.Sprite(img, x=x_center, y=y_center, batch=self.batch)
            # Ensure Anchor is center
            s.image.anchor_x = s.image.width // 2
            s.image.anchor_y = s.image.height // 2
            s.scale = (self.cw * 0.6) / max(s.image.width, s.image.height) # Slightly smaller for looks
            self.sprites.append(s)
            
            # Draw SOS Lines
            current_patterns = self.game.sos_patterns
            
            # Check for new patterns (naive check against currently drawn)
            # We use a set of signatures to track drawn lines
            if not hasattr(self, 'drawn_patterns'):
                self.drawn_patterns = set()
                
            for pattern in current_patterns:
                if pattern not in self.drawn_patterns:
                    self.drawn_patterns.add(pattern)
                    # Pattern is ((r1, c1), (r2, c2), (r3, c3)) sorted
                    coords = list(pattern)
                    # Draw Line from first to last (simplest)
                    # Need to handle wrap visual?
                    # If dist > 1, it's a wrap.
                    
                    p1 = coords[0]
                    p3 = coords[2]
                    
                    x1 = self.margin + p1[1] * self.cw + self.cw/2
                    y1 = (800 - self.margin) - p1[0] * self.cw - self.cw/2
                    x3 = self.margin + p3[1] * self.cw + self.cw/2
                    y3 = (800 - self.margin) - p3[0] * self.cw - self.cw/2
                    
                    # Detect Wrap
                    if abs(p1[0] - p3[0]) > 2 or abs(p1[1] - p3[1]) > 2:
                        # Wrap: Draw dashed/colored or skip line to avoid ugly cross-screen slash
                        pass 
                    else:
                        # Normal Line
                        line = shapes.Line(x1, y1, x3, y3, thickness=4, color=(50, 200, 50), batch=self.batch)
                        line.opacity = 200
                        self.lines.append(line) # Keep ref
            
            self.move_idx += 1
        else:
            pyglet.clock.unschedule(self.update_game)

    def on_draw(self):
        self.clear()
        # Draw background covering whole window
        self.img_bg.blit(0, 0, width=800, height=800)
        self.batch.draw()

# Wrapper to run viewer
def run_replay(file_path):
    # Check if assets exist, else warn
    if not os.path.exists('assets/s.png'):
        print("Warning: Assets not found in ./assets. Replay might fail.")
        
    win = ReplayWindow(file_path)
    pyglet.app.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['plot', 'replay'], help='Mode: plot or replay')
    parser.add_argument('--file', type=str, help='File for replay (json) or log (csv)')
    args = parser.parse_args()
    
    if args.mode == 'plot':
        f = args.file if args.file else 'training_log.csv'
        plot_stats(f)
    elif args.mode == 'replay':
        if not args.file:
            print("Please specify --file replays/replay_X.json")
        else:
            run_replay(args.file)
