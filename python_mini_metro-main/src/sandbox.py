import pygame

from config import framerate, screen_color, screen_height, screen_width,start_with_3_initial_paths
from event.convert import convert_pygame_event
from mediator import Mediator
import json
import datetime
import time
import random
import torch
from torch_geometric.data import Data


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

def display():
    pygame.event.get()
    pygame.display.flip()


mediator.initialize_with_3_paths()
mediator.initialize_action_mapping()
print("len of action map", len(mediator.action_mapping))

action_mask = mediator.available_actions(mask=True)
action_index = mediator.available_actions(mask=False)
print("available actions mask", len(action_mask))
print("available actions mask", action_mask)
print("available actions index", len(action_index))
print("available actions index", action_index)


screen.fill(screen_color)
mediator.render(screen)

# display
for event in pygame.event.get():
    pygame.display.flip()
time.sleep(2)

# update the positions
for t in range(10):

    print(mediator.save_state())     
    break   


    # select random index from index mask where 1 is present
    available_actions = mediator.available_actions(mask=False)
    random_action = random.choice(available_actions)
    print(random_action)
    mediator.select_action(random_action)


    # fast forward time to next time step
    for i in range(250):
        mediator.increment_time(16)

    screen.fill(screen_color)
    mediator.render(screen)
    pygame.event.get()
    pygame.display.flip()

    time.sleep(1)




# for i, pygame_event in enumerate(pygame.event.get()):
#     if pygame_event.type == pygame.QUIT:
#         raise SystemExit
    
#     print(i, pygame_event)
#     pygame.display.flip()




