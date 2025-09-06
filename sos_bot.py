import pyglet
from pyglet import shapes
from pyglet.window import key
from enum import Enum
import random
from greedy_bot import SOSBot   # make sure sos_bot.py is in same folder

# -----------------------
# Constants / Setup
# -----------------------


class GameState(Enum):
    HOME = 0
    PLAYING = 1
    GAME_OVER = 2


window = pyglet.window.Window(fullscreen=True)
game_state = GameState.HOME
line_thickness = 5
space = line_thickness
no_of_cells = 8  # 8 cells
bot_moves_queue = []   # stores multiple moves for the bot
bot_thinking_label = pyglet.text.Label('',
                                       font_name='Arial', font_size=24,
                                       x=window.width//2, y=window.height-50,
                                       anchor_x='center', anchor_y='center')

# grid sizing (same as original)
available_dimension = min(window.width, window.height) * 0.9
total_space_for_gaps = (no_of_cells - 1) * space
grid_size = (available_dimension - total_space_for_gaps) / no_of_cells
total_grid_dimension = no_of_cells * grid_size + total_space_for_gaps

# board and UI lists
board = [[' ' for _ in range(no_of_cells)] for _ in range(no_of_cells)]
players = ['P1', 'Bot']   # Bot replaces P2
SOS = []                  # list of sorted triples already counted
sprites = []              # cell background sprites
line = []                 # line shape objects (kept for reference)

current_player = 0
selected_cell = None
hc = None  # highlight sprite

# grid origin etc.
gridX = window.width // 2 - total_grid_dimension // 2
gridY = window.height // 2 - total_grid_dimension // 2
cellX = gridX
cellY = gridY + total_grid_dimension
cellSize = no_of_cells * grid_size

# batches
batch = pyglet.graphics.Batch()
home_batch = pyglet.graphics.Batch()
overlay_batch = pyglet.graphics.Batch()
highlight_batch = pyglet.graphics.Batch()

# UI elements
home_title = pyglet.text.Label('SOS Game', font_name='Arial', font_size=72,
                               x=window.width//2, y=window.height//2 + 100,
                               anchor_x='center', anchor_y='center', batch=home_batch)
start_button = shapes.Rectangle(window.width//2 - 100, window.height//2 - 50,
                                200, 50, color=(0, 200, 0), batch=home_batch)
start_button_label = pyglet.text.Label('Start Game', font_name='Arial', font_size=24,
                                       x=window.width//2, y=window.height//2 - 25,
                                       anchor_x='center', anchor_y='center', batch=home_batch)

pyglet.resource.path = ['assets']
pyglet.resource.reindex()

S = pyglet.resource.image('s.png')
O = pyglet.resource.image('o.png')
cellBG = pyglet.resource.image('cell.png')
cell_selected = pyglet.resource.image('cell_selected.png')
background = pyglet.resource.image('background.png')

# draw cell backgrounds
for i in range(no_of_cells):
    for j in range(no_of_cells):
        x = gridX + (grid_size + space) * i + space // 2
        y = gridY + (grid_size + space) * j + space // 2
        sprite = pyglet.sprite.Sprite(cellBG, x=x, y=y, batch=batch)
        sprite.scale = (grid_size) / max(cellBG.width, cellBG.height)
        sprites.append(sprite)

# score label (not in batch to avoid pyglet batch/label draw issue)
label = pyglet.text.Label('', font_name='Arial',
                          font_size=20, x=10, y=window.height - 30)

# scores and bot
scores = {'P1': 0, 'Bot': 0}
bot = SOSBot()   # your sos_bot.py; simple greedy/random bot

# -----------------------
# Helpers
# -----------------------


def update_label():
    global label
    if game_state == GameState.PLAYING:
        if current_player == 0:
            # Player turn
            label.text = f"P1: {scores['P1']} | Bot: {scores['Bot']} | P1's Turn"
        else:
            # Bot turn, show thinking instead of instant turn
            if bot_moves_queue:
                label.text = f"P1: {scores['P1']} | Bot: {scores['Bot']} | Bot is thinking..."
            else:
                label.text = f"P1: {scores['P1']} | Bot: {scores['Bot']} | Bot's Turn"
    elif game_state == GameState.GAME_OVER:
        if scores['P1'] > scores['Bot']:
            winner_text = "P1 Wins!"
        elif scores['Bot'] > scores['P1']:
            winner_text = "Bot Wins!"
        else:
            winner_text = "It's a Draw!"
        label.text = f"Game Over! {winner_text}"



def is_board_full():
    for row in board:
        if ' ' in row:
            return False
    return True


def draw_line(x1, y1, x2, y2, player, dotted=False):
    colors = [(0, 183, 239), (237, 28, 36)]
    start_x_screen = gridX + x1 * (grid_size + space) + grid_size // 2
    start_y_screen = gridY + y1 * (grid_size + space) + grid_size // 2
    end_x_screen = gridX + x2 * (grid_size + space) + grid_size // 2
    end_y_screen = gridY + y2 * (grid_size + space) + grid_size // 2

    if dotted:
        dx = end_x_screen - start_x_screen
        dy = end_y_screen - start_y_screen
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            return
        segment_length = 10
        gap_length = 5
        total_segment = segment_length + gap_length
        num_segments = int(length / total_segment)
        for i in range(num_segments):
            seg_start_x = start_x_screen + (dx / length) * (i * total_segment)
            seg_start_y = start_y_screen + (dy / length) * (i * total_segment)
            seg_end_x = start_x_screen + \
                (dx / length) * (i * total_segment + segment_length)
            seg_end_y = start_y_screen + \
                (dy / length) * (i * total_segment + segment_length)
            line.append(shapes.Line(seg_start_x, seg_start_y, seg_end_x, seg_end_y,
                                    thickness=line_thickness, color=colors[player], batch=overlay_batch))
    else:
        line.append(shapes.Line(start_x_screen, start_y_screen, end_x_screen, end_y_screen,
                                thickness=line_thickness, color=colors[player], batch=overlay_batch))


def highlight_cell(y, x):
    global hc
    begX = gridX + x * (grid_size + space)
    begY = gridY + y * (grid_size + space)
    if hc is not None:
        try:
            hc.delete()
        except Exception:
            pass
    hc = pyglet.sprite.Sprite(cell_selected, begX, begY, batch=highlight_batch)
    hc.opacity = 255
    hc.scale = (grid_size) / max(cell_selected.width, cell_selected.height)


def place_symbol(row, col, symbol):
    """Place symbol on board and create sprite (used by both human and bot)."""
    board[row][col] = symbol
    img = S if symbol == 'S' else O
    sprite = pyglet.sprite.Sprite(img,
                                  x=gridX + col *
                                  (grid_size + space) + 0.1 * grid_size,
                                  y=gridY + row *
                                  (grid_size + space) + 0.1 * grid_size,
                                  batch=batch)
    sprite.scale = (0.8 * grid_size) / max(img.width, img.height)
    sprites.append(sprite)

# exact original-style check_win with wrap detection and dedup of SOS triples


def check_win():
    global scores, current_player, SOS

    def is_sos(c, r, dc, dr):
        s1_r, s1_c = r, c

        o_orig_r, o_orig_c = r + dr, c + dc
        s2_orig_r, s2_orig_c = r + 2 * dr, c + 2 * dc

        o_r, o_c = o_orig_r % no_of_cells, o_orig_c % no_of_cells
        s2_r, s2_c = s2_orig_r % no_of_cells, s2_orig_c % no_of_cells

        if board[s1_r][s1_c] == 'S' and board[o_r][o_c] == 'O' and board[s2_r][s2_c] == 'S':
            raw_coords = ((s1_r, s1_c), (o_r, o_c), (s2_r, s2_c))
            sorted_coords = tuple(sorted(raw_coords))

            is_wrapped_sos = (o_orig_r != o_r or o_orig_c != o_c) or \
                             (s2_orig_r != s2_r or s2_orig_c != s2_c)

            return raw_coords, sorted_coords, is_wrapped_sos
        return None, None, False

    found_sos = False
    for r in range(no_of_cells):
        for c in range(no_of_cells):
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                sos_coords_raw, sos_coords_sorted, is_wrapped_sos = is_sos(
                    c, r, dc, dr)
                if sos_coords_raw and sos_coords_sorted not in SOS:
                    SOS.append(sos_coords_sorted)
                    scores[players[current_player]] += 1
                    found_sos = True

                    s1, o, s2 = sos_coords_raw

                    if not is_wrapped_sos:
                        draw_line(s1[1], s1[0], s2[1], s2[0],
                                  current_player, dotted=False)
                    else:
                        draw_line(s1[1], s1[0], o[1], o[0],
                                  current_player, dotted=True)
                        draw_line(o[1], o[0], s2[1], s2[0],
                                  current_player, dotted=True)

    return found_sos


def bot_turn():
    global bot_moves_queue, current_player

    move, letter = bot.choose_move(board)
    bot_moves_queue.append((move, letter))

    # Switch current_player to bot
    current_player = 1
    update_label()  # update immediately

    # Schedule the first move
    pyglet.clock.schedule_once(execute_bot_move, 1.0)


def execute_bot_move(dt):
    global bot_moves_queue, current_player

    if bot_moves_queue:
        (row, col), letter = bot_moves_queue.pop(0)
        place_symbol(row, col, letter)
        scored = check_win()
        update_label()

        if scored and not is_board_full():
            # Schedule next move if bot scored again
            next_move, next_letter = bot.choose_move(board)
            bot_moves_queue.append((next_move, next_letter))
            pyglet.clock.schedule_once(execute_bot_move, 1.0)
        else:
            # End of bot's turn
            current_player = 0
            update_label()  # update now that player can move




def end_game():
    global game_state
    game_state = GameState.GAME_OVER
    update_label()

# -----------------------
# Events
# -----------------------


@window.event
def on_mouse_press(x, y, button, modifiers):
    global game_state, selected_cell, hc
    # Start button on home screen
    if game_state == GameState.HOME:
        if start_button.x <= x <= start_button.x + start_button.width and \
           start_button.y <= y <= start_button.y + start_button.height:
            game_state = GameState.PLAYING
            update_label()
    elif game_state == GameState.PLAYING:
        if gridX <= x <= gridX + cellSize and gridY <= y <= gridY + cellSize:
            cols = int((x - gridX) // (grid_size + space))
            rows = int((y - gridY) // (grid_size + space))
            if board[rows][cols] == ' ':
                selected_cell = (rows, cols)
                highlight_cell(rows, cols)


@window.event
def on_key_press(symbol, modifiers):
    global current_player, hc, selected_cell
    if game_state == GameState.PLAYING and current_player == 0 and selected_cell:
        rows, cols = selected_cell
        if board[rows][cols] == ' ':
            if symbol == key.S:
                place_symbol(rows, cols, 'S')
            elif symbol == key.O:
                place_symbol(rows, cols, 'O')
            else:
                return

            if hc is not None:
                try:
                    hc.delete()
                except Exception:
                    pass
                hc = None

            scored = check_win()
            update_label()
            if not scored and not is_board_full():
                current_player = 1
                bot_turn()
            if is_board_full():
                end_game()


@window.event
def on_draw():
    window.clear()
    if game_state == GameState.HOME:
        background.blit(0, 0, width=window.width, height=window.height)
        home_batch.draw()
    elif game_state == GameState.PLAYING:
        background.blit(0, 0, width=window.width, height=window.height)
        batch.draw()
        overlay_batch.draw()
        highlight_batch.draw()
        label.draw()
    elif game_state == GameState.GAME_OVER:
        background.blit(0, 0, width=window.width, height=window.height)
        batch.draw()
        overlay_batch.draw()
        label.draw()
    elif game_state == GameState.PLAYING:
        background.blit(0, 0, width=window.width, height=window.height)
        batch.draw()
        if hc:
            hc.draw()
        overlay_batch.draw()
        highlight_batch.draw()
        if bot_moves_queue or bot_thinking_label.text:
            bot_thinking_label.draw()
        label.draw()

# -----------------------
# Start
# -----------------------
update_label()
pyglet.app.run()
