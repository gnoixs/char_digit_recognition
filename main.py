import pygame
import numpy as np
import tensorflow as tf
import imageio

model = tf.keras.models.load_model('handwritten.model')

char_digit_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a',
    37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r',
    46: 't'
}

# initialize pygame
pygame.init()
font = pygame.font.Font(None, 30)

# define constants
WINDOW_SIZE = (300, 400)
GRID_SIZE = 28
CELL_SIZE = WINDOW_SIZE[0] // GRID_SIZE
PANEL_SIZE = 50
PANEL_HEIGHT = 100

# create window with white background
screen = pygame.display.set_mode(WINDOW_SIZE)
screen.fill((255, 255, 255))
pygame.display.set_caption("Draw on Grid")

# create blank grid
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# create top panel rectangle
panel_display = pygame.Rect(0, 0, WINDOW_SIZE[0], PANEL_SIZE)

panel_clear = pygame.Rect(0, 50, WINDOW_SIZE[0], PANEL_SIZE)

# default text
default_text = "Draw a letter or digit"

# game loop
running = True
while running:

    # handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.MOUSEMOTION and event.buttons[0]):
            # get cell coordinates based on mouse position
            x, y = event.pos
            col = x // CELL_SIZE
            row = (y - PANEL_HEIGHT) // CELL_SIZE
            # fill cell with black
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                grid[row][col] = 1
        if event.type == pygame.MOUSEBUTTONDOWN and panel_clear.collidepoint(event.pos):
            grid = np.zeros((GRID_SIZE, GRID_SIZE))
            screen.fill((255, 255, 255))

    # draw top panel
    pygame.draw.rect(screen, (200, 200, 200), panel_display)
    pygame.draw.rect(screen, (255, 105, 97), panel_clear)
    text = font.render("Clear", True, (0, 0, 0))
    text_rect = text.get_rect(center=panel_clear.center)
    screen.blit(text, text_rect)

    # draw grid
    black_cells = np.transpose(np.nonzero(grid))
    for cell in black_cells:
        row, col = cell
        rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE + PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (0, 0, 0), rect)

    # display text on top panel

    if np.any(grid):
        # user has drawn on the grid, display different text
        img = np.array(grid, dtype=np.uint8)
        prediction = model.predict(np.array([img]))

        predicted_digit = np.argmax(prediction)
        if predicted_digit in char_digit_map:
            predicted_char = char_digit_map[predicted_digit]
        else:
            predicted_char = "Unknown"

        text = font.render(f"You drew a {predicted_char}", True, (0, 0, 0))

    else:
        # user has not drawn on the grid yet, display default text
        text = font.render(default_text, True, (0, 0, 0))
    text_rect = text.get_rect(center=panel_display.center)
    screen.blit(text, text_rect)

    # update display
    pygame.display.update()

# quit pygame
pygame.quit()
