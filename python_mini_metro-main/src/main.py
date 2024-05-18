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
human_player = True
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


     # save game state in dict
    if mediator.steps % 1000 == 0 and save_states:
        current_state = mediator.save_state()
        if len(game_states) > 0:

            # compute reward and action
            previous_score = game_states[-1]['score']
            current_score = current_state['score']
            game_states[-1]['reward'] = current_score - previous_score

            # Detect new links in paths by comparing adjacency matrices
            new_links = {}
            if len(game_states) > 0:
                for i,color in enumerate(['red','green','blue']):
                    new_links_p = []
                    adj_matrix_prev = game_states[-1]['paths_adj_matrix'][color]
                    adj_matrix_curr = current_state['paths_adj_matrix'][color]
                    
                    for j in range(len(adj_matrix_prev)):
                        for k in range(len(adj_matrix_prev)):
                            if adj_matrix_prev[j][k] != adj_matrix_curr[j][k] and j < k:
                                new_links_p.append((game_states[-1]['station_ids'][j],game_states[-1]['station_ids'][k]))
                    new_links[color] = new_links_p
            game_states[-1]['action'] = new_links # This is just a placeholder/proxy for the actual action space we will define later
            
        game_states.append(current_state)
        
   
   
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
        # add station
        if mediator.steps >= 100 and mediator.steps < 200:
            mediator.agent_add_station_to_path(mediator.paths[0],mediator.stations[0])


        if enable_graphics:
            for pygame_event in pygame.event.get():
                if pygame_event.type == pygame.QUIT:
                    raise SystemExit
            pygame.display.flip()
            

            raise SystemExit
        else:
            event = convert_pygame_event(pygame_event)
            mediator.react(event)
            # mediator.react(game_states)

    pygame.display.flip()
    
pygame.time.delay(1000) # 2초 딜레이 (ms기준)
pygame.quit()

# Save game states to a JSON file
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
filename = f"{current_datetime}.json"
with open(filename, 'w') as file:
    json.dump(game_states, file)