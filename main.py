import pyglet 
from pyglet.window import mouse

window = pyglet.window.Window(visible=False, fullscreen=True)
image = pyglet.image.load(r'C:/Users/moham/Documents/PROGRAMMING/web_dev/image/heart.png')
label = pyglet.text.Label('Hello, world',
                          font_name='Comic Sans MS',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')

window.set_visible()
@window.event
def on_key_press(symbol, modifiers):
    print('A key was pressed')

@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == mouse.LEFT:
        print('The left mouse button was pressed.')


# event_logger = pyglet.window.event.WindowEventLogger()
# window.push_handlers(event_logger)

# music = pyglet.resource.media('aud.wav')
# music.play()

@window.event
def on_draw():
    window.clear()
    label.draw()
    for i in range(0, window.width, image.width + 10):
        for j in range(0, window.height, image.height + 10):
            image.blit(i, j)

pyglet.app.run()