# import os
# import subprocess
# # Define the combinations of epsilon and random_seed
# num_actions = [24,12]
# epsilons = [1,0,0.5] #0
# random_seeds = list(range(30))# 12, 0 8 ~ / epsilon=1 instead of 0 ?!


# # Template for config.py
# config_template = """
# from geometry.type import ShapeType

# # game
# framerate = 60

# # screen
# screen_width = 640
# screen_height = 360
# screen_color = (255, 255, 255)

# # station
# num_stations = 10
# station_size = 15
# station_capacity = 12
# station_color = (0, 0, 0)
# station_shape_type_list = [
#     ShapeType.RECT,
#     ShapeType.CIRCLE,
#     ShapeType.TRIANGLE,
#     ShapeType.CROSS,
# ]
# station_passengers_per_row = 4

# # passenger
# passenger_size = 3
# passenger_color = (128, 128, 128)
# passenger_spawning_start_step = 1
# passenger_spawning_interval_step = 10 * framerate
# passenger_display_buffer = 3 * passenger_size

# # metro
# num_metros = 4
# metro_size = 15
# metro_color = (200, 200, 200)
# metro_capacity = 6
# metro_speed_per_ms = 150 / 1000  # pixels / ms
# metro_passengers_per_row = 3

# # path
# num_paths = 3
# path_width = 7
# path_order_shift = 10

# # button
# button_color = (180, 180, 180)
# button_size = 15

# # path button
# path_button_buffer = 20
# path_button_dist_to_bottom = 50
# path_button_start_left = 500
# path_button_cross_size = 25
# path_button_cross_width = 5

# # text
# score_font_size = 30
# score_display_coords = (10, 10)

# # gameover text
# gameover_font_size = 60
# gameover_text = "Game Over"
# gameover_text_color = (255, 0, 0)
# gameover_text_coords = (screen_width // 2, screen_height // 2)
# gameover_text_center = True

# #startwith 3initial paths
# start_with_3_initial_paths = True

# #other settings
# # Configurations
# running = True
# save_states = False
# human_player = True
# enable_graphics = True
# greedy_agent = True
# a3c_agent = False

# num_action = {num_action}
# epsilon = {epsilon}
# random_seed = {random_seed}
# """

# # Path to the main script
# main_script = 'main.py'

# # Function to run the main script
# def run_script():
#     os.system(f'python {main_script}')

# # Loop through each combination of epsilon and random_seed
# for num_action in num_actions:
#     for epsilon in epsilons:
#         for random_seed in random_seeds:
#             # Create the new config.py content
#             config_content = config_template.format(epsilon=epsilon, random_seed=random_seed, num_action=num_action)
            
#             # Write the new config.py
#             with open('config.py', 'w') as config_file:
#                 config_file.write(config_content)
            
#             print('configs: ',num_action, epsilon, random_seed)
#             # Run the main script
#             run_script()
#             log_file = f'output_num_action_{num_action}_epsilon_{epsilon}_seed_{random_seed}.log'
            
#             # Run the main script and capture the output
#             with open(log_file, 'w') as output_file:
#                 process = subprocess.run(['python', main_script], stdout=output_file, stderr=output_file)
            
#             print(f'configs: {num_action}, {epsilon}, {random_seed} - log saved to {log_file}')

# print("all actions completed.")

# num_actions = [12]
# epsilons = [0]
# random_seeds = range(10)

# for num_action in num_actions:
#     for epsilon in epsilons:
#         for random_seed in random_seeds:
#             # Create the new config.py content
#             config_content = config_template.format(epsilon=epsilon, random_seed=random_seed, num_action=num_action)
            
#             # Write the new config.py
#             with open('config.py', 'w') as config_file:
#                 config_file.write(config_content)
            
#             print('configs: ',num_action, epsilon, random_seed)
#             # Run the main script
#             run_script()

# print("12 actions completed.")


import os
import subprocess
import pandas as pd

# Define the combinations of epsilon and random_seed
num_actions = [24,12]
epsilons = [1,0,0.5]
random_seeds = list(range(30))

# num_actions = [24, 12]
# epsilons = [1, 0, 0.5]
# random_seeds = list(range(30))

# Template for config.py
config_template = """
from geometry.type import ShapeType

# game
framerate = 60

# screen
screen_width = 640
screen_height = 360
screen_color = (255, 255, 255)

# station
num_stations = 10
station_size = 15
station_capacity = 12
station_color = (0, 0, 0)
station_shape_type_list = [
    ShapeType.RECT,
    ShapeType.CIRCLE,
    ShapeType.TRIANGLE,
    ShapeType.CROSS,
]
station_passengers_per_row = 4

# passenger
passenger_size = 3
passenger_color = (128, 128, 128)
passenger_spawning_start_step = 1
passenger_spawning_interval_step = 10 * framerate
passenger_display_buffer = 3 * passenger_size

# metro
num_metros = 4
metro_size = 15
metro_color = (200, 200, 200)
metro_capacity = 6
metro_speed_per_ms = 150 / 1000  # pixels / ms
metro_passengers_per_row = 3

# path
num_paths = 3
path_width = 7
path_order_shift = 10

# button
button_color = (180, 180, 180)
button_size = 15

# path button
path_button_buffer = 20
path_button_dist_to_bottom = 50
path_button_start_left = 500
path_button_cross_size = 25
path_button_cross_width = 5

# text
score_font_size = 30
score_display_coords = (10, 10)

# gameover text
gameover_font_size = 60
gameover_text = "Game Over"
gameover_text_color = (255, 0, 0)
gameover_text_coords = (screen_width // 2, screen_height // 2)
gameover_text_center = True

#startwith 3initial paths
start_with_3_initial_paths = True

#other settings
# Configurations
running = True
save_states = False
human_player = True
enable_graphics = True
greedy_agent = True
a3c_agent = False

num_action = {num_action}
epsilon = {epsilon}
random_seed = {random_seed}
"""

# Path to the main script
main_script = 'main.py'

def run_script():
    os.system(f'python {main_script}')



# Loop through each combination of epsilon and random_seed
for num_action in num_actions:
    for epsilon in epsilons:
        for random_seed in random_seeds:
            # Create the new config.py content
            config_content = config_template.format(epsilon=epsilon, random_seed=random_seed, num_action=num_action)
            
            # Write the new config.py
            with open('config.py', 'w') as config_file:
                config_file.write(config_content)
            
            print('configs: ',num_action, epsilon, random_seed)
            # Run the main script
            run_script()
            log_file = f'output_num_action_{num_action}_epsilon_{epsilon}_seed_{random_seed}.log'
            
         # Run the main script and capture the output
            with open(log_file, 'w') as output_file:
                process = subprocess.run(['python', main_script], stdout=output_file, stderr=output_file)
            
            print(f'configs: {num_action}, {epsilon}, {random_seed} - log saved to {log_file}')


print("All actions completed. Results saved to results.csv.")
