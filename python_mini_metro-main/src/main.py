import pygame

from config import framerate, screen_color, screen_height, screen_width,start_with_3_initial_paths
from event.convert import convert_pygame_event
from mediator import Mediator
import json
import datetime

# init
pygame.init()

# settings
flags = pygame.SCALED

# game constants initialization
screen = pygame.display.set_mode((screen_width, screen_height), flags, vsync=1)
clock = pygame.time.Clock()

mediator = Mediator()
game_states = []

# Configurations
running = True
save_states = True
enable_graphics = True


while running:
    dt_ms = clock.tick(framerate)
    if start_with_3_initial_paths:
        mediator.initialize_with_3_paths()
        start_with_3_initial_paths = False

    mediator.increment_time(dt_ms)
    if enable_graphics:
        screen.fill(screen_color)
        mediator.render(screen)
    running = mediator.is_gameover()
    mediator.matrix_state()

   
    for pygame_event in pygame.event.get():
        if pygame_event.type == pygame.QUIT:
            raise SystemExit
        else:
            event = convert_pygame_event(pygame_event)
            mediator.react(event)
    pygame.display.flip()
    
            


pygame.time.delay(1000) # 2초 딜레이 (ms기준)
pygame.quit()

