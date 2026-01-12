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
from game_logic import SOSGameBase  
# Note: original game_logic.py has SOSGame. 
# We need to replicate the visual logic from sos_1v1.py but automated.

# ---------------- PLOTTING ----------------
def plot_stats(log_file='training_log.csv'):
    if not os.path.exists(log_file):
        print(f"No log file found at {log_file}")
        return
        
    df = pd.read_csv(log_file)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Loss
    ax[0].plot(df['Iteration'], df['PolicyLoss'], label='Policy Loss')
    ax[0].plot(df['Iteration'], df['ValueLoss'], label='Value Loss')
    ax[0].set_title('Training Losses')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)
    
    # Moves
    ax[1].plot(df['Iteration'], df['AvgMoves'], label='Avg Moves', color='green')
    ax[1].set_title('Average Game Length')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Moves')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
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
        self.game = SOSGameBase() # We need a headless version for logic, visual separate?
        # Actually easier to just draw the grid and S/O
        
        self.grid_cells = [[' ' for _ in range(8)] for _ in range(8)]
        self.sprites = []
        self.lines = []
        self.batch = pyglet.graphics.Batch()
        
        # S and O images (Assumed in assets folder relative closely, or absolute)
        # Assuming current dir is PyGames1
        pyglet.resource.path = ['assets']
        pyglet.resource.reindex()
        self.img_s = pyglet.resource.image('s.png')
        self.img_o = pyglet.resource.image('o.png')
        self.img_bg = pyglet.resource.image('background.png')
        self.img_cell = pyglet.resource.image('cell.png')
        
        # Grid calc
        self.margin = 50
        self.cw = (800 - 2*self.margin) / 8
        
        # Schedule update
        pyglet.clock.schedule_interval(self.update_game, 0.5) # 2 moves per second

    def update_game(self, dt):
        if self.move_idx < len(self.moves):
            action = self.moves[self.move_idx]
            
            # Decode action
            is_o = action >= 64
            idx = action - 64 if is_o else action
            r = idx // 8
            c = idx % 8
            
            letter = 'O' if is_o else 'S'
            self.grid_cells[r][c] = letter
            
            # Graphic
            x = self.margin + c * self.cw + self.cw/2
            y = 800 - (self.margin + r * self.cw + self.cw/2) # Inverted Y usually?
            # Pyglet Y is 0 at bottom.
            # Row 0 is Top usually in matrix? 
            # game_logic uses 0,0. Let's assume 0,0 is top-left for matrix
            y = 800 - self.margin - r * self.cw - self.cw/2
            
            img = self.img_o if is_o else self.img_s
            s = pyglet.sprite.Sprite(img, x=x, y=y, batch=self.batch)
            s.image.anchor_x = s.width // 2
            s.image.anchor_y = s.height // 2
            s.scale = (self.cw * 0.8) / max(s.width, s.height)
            self.sprites.append(s)
            
            self.move_idx += 1
        else:
            pyglet.clock.unschedule(self.update_game)

    def on_draw(self):
        self.clear()
        self.img_bg.blit(0, 0, width=800, height=800)
        
        # Draw Grid
        # ... (Simplified grid)
        
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
