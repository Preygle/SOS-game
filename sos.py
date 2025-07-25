import pyglet
from pyglet import shapes
from pyglet.window import key
from enum import Enum

class GameState(Enum):
    HOME = 0
    PLAYING = 1
    GAME_OVER = 2

window = pyglet.window.Window(fullscreen=True)
game_state = GameState.HOME
line_thickness = 5
space = line_thickness
no_of_cells = 8 #8 cells

# Calculate grid_size dynamically based on window dimensions
# Aim for the grid to take up 90% of the smaller window dimension.
# This will leave a 5% margin on each side.
available_dimension = min(window.width, window.height) * 0.9
total_space_for_gaps = (no_of_cells - 1) * space
grid_size = (available_dimension - total_space_for_gaps) / no_of_cells

# Calculate total grid dimensions (including cells and spaces)
total_grid_dimension = no_of_cells * grid_size + total_space_for_gaps

board = [[' ' for i in range(no_of_cells)] for i in range(no_of_cells)]
players = ['P1', 'P2']
SOS = []
sprites = []
line = []

current_player = 0
hc = None

# Calculate the bottom-left corner of the entire grid
gridX = window.width // 2 - total_grid_dimension // 2
gridY = window.height // 2 - total_grid_dimension // 2

# These are for drawing the grid lines.
# cellX is the left edge of the grid.
# cellY is the top edge of the grid.
cellX = gridX
cellY = gridY + total_grid_dimension

# cellSize is the total dimension of all cells combined, without spaces.
cellSize = no_of_cells * grid_size

batch = pyglet.graphics.Batch()
home_batch = pyglet.graphics.Batch()

# Home Screen Elements
home_title = pyglet.text.Label('SOS Game', font_name='Arial', font_size=72,
                               x=window.width//2, y=window.height//2 + 100,
                               anchor_x='center', anchor_y='center', batch=home_batch)
start_button = shapes.Rectangle(window.width//2 - 100, window.height//2 - 50, 200, 50, color=(0, 200, 0), batch=home_batch)
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

for i in range(no_of_cells):
    for j in range(no_of_cells):
        x = gridX + (grid_size + space) * i + space // 2
        y = gridY + (grid_size + space) * j + space // 2
        sprite = pyglet.sprite.Sprite(cellBG, x=x, y=y, batch=batch)
        sprite.scale = (grid_size ) / max(cellBG.width, cellBG.height)
        sprites.append(sprite)


selected_cell = None
keys = key.KeyStateHandler()



scores = {'P1': 0, 'P2': 0}
label = pyglet.text.Label('', font_name='Arial', font_size=20, x=10, y=window.height - 30, batch=batch)

def update_label():
    global label
    if game_state == GameState.PLAYING:
        label.text = f"P1: {scores['P1']} | P2: {scores['P2']} P{current_player + 1} Turn"
    elif game_state == GameState.GAME_OVER:
        if scores['P1'] > scores['P2']:
            winner_text = "P1 Wins!"
        elif scores['P2'] > scores['P1']:
            winner_text = "P2 Wins!"
        else:
            winner_text = "It's a Draw!"
        label.text = f"Game Over! {winner_text}"

update_label()

keys = key.KeyStateHandler()
window.push_handlers(keys)

# for i in range(0,no_of_cells + 1):
#     #horizontal Lines
#     line.append(shapes.Line(cellX + grid_size * i, cellY, cellX + grid_size * i, cellY - cellSize, thickness=line_thickness, color=(255, 255, 255), batch=batch))
#     #Vertical lines
#     line.append(shapes.Line(cellX, cellY - grid_size * i, cellX + cellSize, cellY - grid_size * i, thickness=line_thickness, color=(255, 255, 255), batch=batch))


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
                sos_coords_raw, sos_coords_sorted, is_wrapped_sos = is_sos(c, r, dc, dr)
                if sos_coords_raw and sos_coords_sorted not in SOS:
                    SOS.append(sos_coords_sorted)
                    scores[players[current_player]] += 1
                    found_sos = True
                    
                    s1, o, s2 = sos_coords_raw

                    if not is_wrapped_sos:
                        draw_line(s1[1], s1[0], s2[1], s2[0], current_player, dotted=False)
                    else:
                        draw_line(s1[1], s1[0], o[1], o[0], current_player, dotted=True)
                        draw_line(o[1], o[0], s2[1], s2[0], current_player, dotted=True)

    return found_sos


overlay_batch = pyglet.graphics.Batch()

def draw_line(x1, y1, x2, y2, player, dotted=False):
   global line
   start_x_screen = gridX + x1 * (grid_size + space) + grid_size // 2
   start_y_screen = gridY + y1 * (grid_size + space) + grid_size // 2
   end_x_screen = gridX + x2 * (grid_size + space) + grid_size // 2
   end_y_screen = gridY + y2 * (grid_size + space) + grid_size // 2
   colors = [(0, 183, 239), (237, 28, 36)] 
   
   if dotted:
        dx = end_x_screen - start_x_screen
        dy = end_y_screen - start_y_screen
        length = (dx**2 + dy**2)**0.5
        if length == 0: return

        segment_length = 10
        gap_length = 5
        total_segment = segment_length + gap_length

        num_segments = int(length / total_segment)

        for i in range(num_segments):
            seg_start_x = start_x_screen + (dx / length) * (i * total_segment)
            seg_start_y = start_y_screen + (dy / length) * (i * total_segment)
            seg_end_x = start_x_screen + (dx / length) * (i * total_segment + segment_length)
            seg_end_y = start_y_screen + (dy / length) * (i * total_segment + segment_length)
            line.append(shapes.Line(seg_start_x, seg_start_y, seg_end_x, seg_end_y, thickness=line_thickness, color=colors[player], batch=overlay_batch))
   else:
       line.append(shapes.Line(start_x_screen, start_y_screen, end_x_screen, end_y_screen, thickness=line_thickness, color=colors[player], batch=overlay_batch))


highlight_batch = pyglet.graphics.Batch()
def highlight_cell(y, x):
    global hc
    begX = gridX + x * (grid_size + space)
    begY = gridY + y * (grid_size + space)

    hc = pyglet.sprite.Sprite(cell_selected,
                                  begX,
                                  begY
                                 , batch=highlight_batch)
    hc.opacity = 255
    hc.scale = (grid_size) / max(cell_selected.width, cell_selected.height)
    #sprites.append(sprite)
    

     
@window.event
def on_mouse_press(x, y, button, modifiers):
    
    global selected_cell, hc
    if gridX <= x <= gridX + cellSize and gridY <= y <= gridY + cellSize:
        cols = int((x - gridX) // (grid_size + space))
        rows = int((y - gridY) // (grid_size + space))
        if board[rows][cols] == ' ':
                selected_cell = (rows, cols)
                highlight_cell(rows, cols)

        
def is_board_full():
    for rows in board:
        if ' ' in rows:
            return False
    return True

@window.event
def on_key_press(symbol, modifiers):
    global sprites, current_player, hc
    if selected_cell:
        rows, cols = selected_cell
        if board[rows][cols] == ' ':
            if symbol == key.S:
                board[rows][cols] = 'S'
                sprite = pyglet.sprite.Sprite(S,
                                              x=gridX + cols * (grid_size + space) + 0.1 * grid_size,
                                              y=gridY + rows * (grid_size + space) + 0.1 * grid_size, batch=batch)

                sprite.scale = (0.8 * grid_size) / max(S.width, S.height)
                sprites.append(sprite)
            elif symbol == key.O:
                board[rows][cols] = 'O'
                sprite = pyglet.sprite.Sprite(O,
                                              x=gridX + cols * (grid_size + space) + 0.1 * grid_size,
                                              y=gridY + rows * (grid_size + space) + 0.1 * grid_size, batch=batch)
                sprite.scale = (0.8 * grid_size) / max(O.width, O.height)
                sprites.append(sprite)

            if hc is not None:
                hc.delete()
                hc = None

            scored = check_win()  # Store result to determine whether to switch player
            if scored:
                print(
                    f"{players[current_player]} scored! New Score: P1={scores['P1']} P2={scores['P2']}")
            else:
                # Switch player only if no score
                current_player = (current_player + 1) % 2
            if is_board_full():
                if scores['P1'] > scores['P2']:
                    winner_text = "P1 Wins!"
                elif scores['P2'] > scores['P1']:
                    winner_text = "P2 Wins!"
                else:
                    winner_text = "It's a Draw!"

                print("Game Over!", winner_text)
                game_state = GameState.GAME_OVER
            update_label()
            print(SOS)


@window.event
def on_draw():
    window.clear()
    if game_state == GameState.HOME:
        background.blit(0, 0, width=window.width, height=window.height)
        home_batch.draw()
    elif game_state == GameState.PLAYING:
        background.blit(0, 0, width=window.width, height=window.height)
        batch.draw()    
        if hc:
             hc.draw()
        if sprites:
            for sprite in sprites:
                sprite.draw()
        overlay_batch.draw()
        highlight_batch.draw()
        
    elif game_state == GameState.GAME_OVER:
        background.blit(0, 0, width=window.width, height=window.height)
        batch.draw()
        

@window.event
def on_mouse_press(x, y, button, modifiers):
    global game_state, selected_cell, hc
    if game_state == GameState.HOME:
        if start_button.x <= x <= start_button.x + start_button.width and \
           start_button.y <= y <= start_button.y + start_button.height:
            game_state = GameState.PLAYING
    elif game_state == GameState.PLAYING:
        if gridX <= x <= gridX + cellSize and gridY <= y <= gridY + cellSize:
            cols = int((x - gridX) // (grid_size + space))
            rows = int((y - gridY) // (grid_size + space))
            if board[rows][cols] == ' ':
                    selected_cell = (rows, cols)
                    highlight_cell(rows, cols)

pyglet.app.run()