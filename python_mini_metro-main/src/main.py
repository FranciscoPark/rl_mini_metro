import pygame

from config import framerate, screen_color, screen_height, screen_width,start_with_3_initial_paths
from event.convert import convert_pygame_event
from mediator import Mediator
from rl_agent import Agent

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
human_player = False
enable_graphics = True
# assert screen is shown for human players


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

   
    # react to user interaction
    if human_player:
        for pygame_event in pygame.event.get():
            if pygame_event.type == pygame.QUIT:

                raise SystemExit
            else:
                event = convert_pygame_event(pygame_event)
                mediator.react(event)

        pygame.display.flip()
    else:
        
        if mediator.steps%1000 == 20:

            # mediator.matrix_state()
            state = mediator.save_state()
            agent = Agent(state, 0.1) # input state and Exploration rate
            action = agent.choose_action()
            mediator.agent_add_station_to_path(action[0],action[1])
                    
        if enable_graphics:
            for pygame_event in pygame.event.get():
                if pygame_event.type == pygame.QUIT:
                    raise SystemExit
            pygame.display.flip()
            

print("-------- Got here")
pygame.time.delay(1000) # 2초 딜레이 (ms기준)
pygame.quit()
