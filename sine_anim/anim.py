
import pygame
import math
import sys
import os

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 700 # Increased height for sliders
BG_COLOR = (30, 30, 30)
FPS = 60
FONT = pygame.font.SysFont("Arial", 16)

# Default Animation settings
AMPLITUDE = 18.67
FREQUENCY = 1.82
SPEED = 0.17
NUM_CHUNKS = 37

# Setup screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sine Wave Animation")
clock = pygame.time.Clock()

# Load image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "button2.png")

try:
    image = pygame.image.load(image_path).convert_alpha()
except FileNotFoundError:
    print(f"Error: {image_path} not found.")
    sys.exit()

img_rect = image.get_rect()
img_width, img_height = img_rect.size

# Center position for the image (shifted up to make room for UI)
# Sliders start at y=500, so image should be centered above that.
IMAGE_AREA_HEIGHT = 500
center_x = (WIDTH - img_width) // 2
center_y = (IMAGE_AREA_HEIGHT - img_height) // 2


class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.label = label
        self.dragging = False
        self.handle_width = 10
        self.handle_rect = pygame.Rect(0, y - 5, self.handle_width, h + 10) # x will be updated
        self.update_handle_pos()

    def update_handle_pos(self):
        # Map value to x position
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        self.handle_rect.centerx = self.rect.x + (self.rect.width * ratio)

    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.handle_rect.collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                    self.dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.dragging = False

        if self.dragging:
            # Clamp x to slider width
            x = max(self.rect.x, min(mouse_pos[0], self.rect.right))
            ratio = (x - self.rect.x) / self.rect.width
            self.val = self.min_val + (self.max_val - self.min_val) * ratio
            self.update_handle_pos()

    def draw(self, surface):
        # Draw track
        pygame.draw.rect(surface, (100, 100, 100), self.rect)
        # Draw handle
        pygame.draw.rect(surface, (200, 200, 200), self.handle_rect)
        
        # Draw label and value
        label_surf = FONT.render(f"{self.label}: {self.val:.2f}", True, (255, 255, 255))
        surface.blit(label_surf, (self.rect.x, self.rect.y - 20))

def main():
    global AMPLITUDE, FREQUENCY, SPEED, NUM_CHUNKS

    # Initialize sliders
    slider_y_start = 550 # Adjusted to be lower to give more space for image
    sliders = [
        Slider(50, slider_y_start, 300, 10, 0, 50, AMPLITUDE, "Amplitude"),
        Slider(50, slider_y_start + 50, 300, 10, 0.1, 10.0, FREQUENCY, "Frequency"),
        Slider(400, slider_y_start, 300, 10, 0.0, 1.0, SPEED, "Speed"),
        Slider(400, slider_y_start + 50, 300, 10, 1, 100, NUM_CHUNKS, "Chunks")
    ]

    offset = 0
    running = True

    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        # Update sliders
        for slider in sliders:
            slider.update(events)

        # Update params from sliders
        AMPLITUDE = sliders[0].val
        FREQUENCY = sliders[1].val
        SPEED = sliders[2].val
        NUM_CHUNKS = int(sliders[3].val)

        # Update animation offset
        offset += SPEED

        # Drawing
        screen.fill(BG_COLOR)

        # Draw sliders
        for slider in sliders:
            slider.draw(screen)

        # Calculate strip width
        # Ensure NUM_CHUNKS is at least 1 to avoid division by zero
        current_num_chunks = max(1, NUM_CHUNKS)
        msg_width = max(1, img_width // current_num_chunks)

        # Draw the image in strips
        startX = center_x
        startY = center_y

        # Draw strips
        for i in range(current_num_chunks):
            x_pos = i * msg_width
            if x_pos >= img_width: 
                break # Stop if we've drawn past the image width
            
            w = min(msg_width, img_width - x_pos)
            
            strip_rect = pygame.Rect(x_pos, 0, w, img_height)
            strip_surface = image.subsurface(strip_rect)
            
            # The original code used x (pixel position) for the sine wave calculation.
            # To make the FREQUENCY slider more intuitive with a range of 0.1-10,
            # we scale the x_pos by a small factor (e.g., 0.01) so that FREQUENCY
            # acts on a more normalized range.
            y_shift = AMPLITUDE * math.sin(FREQUENCY * 0.01 * x_pos + offset) 

            screen.blit(strip_surface, (startX + x_pos, startY + y_shift))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
