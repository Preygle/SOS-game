from PIL import Image
import os

def resize_image(filename):
    path = os.path.join('assets', filename)
    if not os.path.exists(path): return

    img = Image.open(path)
    # Resize with NEAREST to keep pixel look
    # Target: 300x80 (matches code)
    img = img.resize((300, 80), Image.Resampling.NEAREST)
    img.save(path)
    print(f"Resized {filename} to 300x80")

resize_image('button.png')
resize_image('button_pressed.png')
