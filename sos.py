import pyglet
from pyglet import shapes
from pyglet.window import key

window = pyglet.window.Window(1080, 1080, fullscreen=True)
line_thickness = 5
space = line_thickness
grid_size = 60 + 4 * space #60
no_of_cells = 8 #8 cells
board = [[' ' for i in range(no_of_cells)] for i in range(no_of_cells)]
players = ['P1', 'P2']
SOS = []
sprites = []
line = []

current_player = 0
hc = None
gridX = window.width // 2 - grid_size * no_of_cells // 2
gridY = cellY = window.height // 2 - grid_size * no_of_cells // 2

cellX = window.width // 2 - grid_size * no_of_cells // 2
cellY = window.height // 2 + grid_size * no_of_cells // 2
cellSize = grid_size * no_of_cells

batch = pyglet.graphics.Batch()

pyglet.resource.path = ['assets']
pyglet.resource.reindex()

S = pyglet.resource.image('s.png')
O = pyglet.resource.image('o.png')
cellBG = pyglet.resource.image('cell.png')
cell_selected = pyglet.resource.image('cell_selected.png')

for i in range(no_of_cells):
    for j in range(no_of_cells):
        x = gridX + (grid_size + space) * i + space // 2
        y = gridY + (grid_size + space) * j + space // 2
        sprite = pyglet.sprite.Sprite(cellBG, x=x, y=y, batch=batch)
        sprite.scale = (grid_size ) / max(cellBG.width, cellBG.height)
        sprites.append(sprite)


selected_cell = None
keys = key.KeyStateHandler()

square = shapes.Rectangle(gridX, gridY, cellSize + no_of_cells * space, cellSize + no_of_cells * space, color=(255, 255, 255), batch=batch)

scores = {'P1': 0, 'P2': 0}
label = pyglet.text.Label(f"P1: {scores['P1']} | P2: {scores['P2']} P{current_player + 1} Turn",
                          font_name='Arial', font_size=20, x=10, y=window.height - 30, batch=batch)

keys = key.KeyStateHandler()
window.push_handlers(keys)

for i in range(0,no_of_cells + 1):
    #horizontal Lines
    line.append(shapes.Line(cellX + grid_size * i, cellY, cellX + grid_size * i, cellY - cellSize, thickness=line_thickness, color=(255, 255, 255), batch=batch))
    #Vertical lines
    line.append(shapes.Line(cellX, cellY - grid_size * i, cellX + cellSize, cellY - grid_size * i, thickness=line_thickness, color=(255, 255, 255), batch=batch))


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
   start_x = gridX + x1 * (grid_size + space) + grid_size // 2
   start_y = gridY + y1 * (grid_size + space) + grid_size // 2
   end_x = gridX + x2 * (grid_size + space) + grid_size // 2
   end_y = gridY + y2 * (grid_size + space) + grid_size // 2
   colors = [(0, 183, 239), (237, 28, 36)] 
   
   if dotted:
        dx = end_x - start_x
        dy = end_y - start_y
        length = (dx**2 + dy**2)**0.5
        if length == 0: return

        segment_length = 10
        gap_length = 5
        total_segment = segment_length + gap_length

        num_segments = int(length / total_segment)

        for i in range(num_segments):
            seg_start_x = start_x + (dx / length) * (i * total_segment)
            seg_start_y = start_y + (dy / length) * (i * total_segment)
            seg_end_x = start_x + (dx / length) * (i * total_segment + segment_length)
            seg_end_y = start_y + (dy / length) * (i * total_segment + segment_length)
            line.append(shapes.Line(seg_start_x, seg_start_y, seg_end_x, seg_end_y, thickness=line_thickness, color=colors[player], batch=overlay_batch))
   else:
       line.append(shapes.Line(start_x, start_y, end_x, end_y, thickness=line_thickness, color=colors[player], batch=overlay_batch))


highlight_batch = pyglet.graphics.Batch()
def highlight_cell(x, y):
    global hc
    begX = gridX + x * (grid_size + space)
    begY = gridY + y * (grid_size + space)

    hc = pyglet.sprite.Sprite(cell_selected,
                                  begY,
                                  begX
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
                label.text = f"Game Over! {winner_text}"
            else:
                label.text = f"P1: {scores['P1']} | P2: {scores['P2']} P{current_player + 1} Turn"
            print(SOS)


@window.event
def on_draw():
    window.clear()
    batch.draw()    
    if hc:
         hc.draw()
    if sprites:
        for sprite in sprites:
            sprite.draw()
    overlay_batch.draw()
    highlight_batch.draw()
pyglet.app.run()