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
#station_capacity = 12
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
num_action = 12
epsilon =0.5