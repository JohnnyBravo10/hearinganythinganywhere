import numpy as np
import config
import trace1 as G
import rooms.dataset as dataset

import torch

"""
Importing this document automatically loads data from the classroom dataset
"""

speed_of_sound = 343

"""
Locations of all surfaces in Meters
"""

#inches = 0.0254

max_x = 7.1247
max_y = 7.9248
max_z = 2.7432

rear_wall = G.Surface(np.array([[16.98386952,-12.74455183,0], [16.98386952,-12.74455183,6.76461222], [0,-12.74455183,0]]))

front_wall = G.Surface(np.array([[0,0,0], [0,0,6.76461222], [16.98386952,0,0]]))


floor = G.Surface(np.array([[-3.40656751, -12.74455183, 0], [-3.40656751, 1.05461696, 0], [16.98386952,-12.74455183,0]]))


ceiling = G.Surface(np.array([[0, 0, 6.76461222], [0, -12.74455183, 6.76461222], [16.98386952,0,6.76461222]]))

left_wall = G.Surface(np.array([[0, 0, 0], [0, 0, 6.76461222], [0,-12.74455183,0]]))

right_wall = G.Surface(np.array([[16.98386952, 0, 0], [16.98386952, 0, 6.76461222], [16.98386952,-12.74455183,0]]))

################################################àà
right_table_2 = G.Surface(np.array([[6.80, -1.30, 0.73], [6.80, -2.90, 0.73], [7.60,-1.30,0.73]]))

right_table_1 = G.Surface(np.array([[6.80, -3.90, 0.73], [6.80, -5.50, 0.73], [7.60,-3.90,0.73]]))

main_table = G.Surface(np.array([[4.40, -6, 0.73], [6, -6, 0.73], [4.40,-6.80,0.73]]))

####################################################

right_door = G.Surface(np.array([[16.8, -9.1610978, 3.2440766], [16.8, -11.59951729, 3.2440766], [16.8,-9.1610978,0]]))

left_door_1 = G.Surface(np.array([[0, -7, 0], [0, -9, 0], [0,-7, 2]]))

left_door_2 = G.Surface(np.array([[0, -3.10, 0], [0, -5.60, 0], [0,-3.10, 2]]))

left_door_3 = G.Surface(np.array([[0, -0.30, 0], [0, -2.10, 0], [0,-0.30, 2]]))

####################################################

monitor = G.Surface(np.array([[6.85, -4.35, 0.90], [6.85, -5.05, 0.90], [6.85,-4.35,1.30]]))

big_panel = G.Surface(np.array([[16.8, -4, 2], [16.8, -4, 4.50], [16.8,-8.50,2]]))

green_screen_1 = G.Surface(np.array([[16.98386952, 0, 0], [16.8, 0, 2.40], [16.8,-4.20,0]]))

green_screen_2 = G.Surface(np.array([[16.98386952, 0, 0], [13.50, 0, 0], [16.98386952, 0, 2.40]]))




walls = [rear_wall, front_wall, floor, ceiling, left_wall, right_wall]
tables = [right_table_2, right_table_1, main_table]
doors = [left_door_1, left_door_2, left_door_3, right_door]
panels = [monitor, big_panel, green_screen_1, green_screen_2]

base_surfaces = walls+tables+doors+panels


"""
Train and Test Split
"""

train_indices = np.array([0])

valid_indices = np.array([])


#Speaker xyz estimated from 12-point TOA, inside speaker, 8.5cm away from manual measurement.
BaseDataset = dataset.Dataset(
   load_dir = config.nottingham_S3_path,
   speaker_xyz= np.array([5.78, 2.71, 1.40]), 
   all_surfaces = base_surfaces,
   speed_of_sound = speed_of_sound,
   default_binaural_listener_forward = np.array([0,1,0]),
   default_binaural_listener_left = np.array([-1,0,0]),
   parallel_surface_pairs=[[0,1], [2,3], [4,5]],
   train_indices = train_indices,
   valid_indices = valid_indices,
   max_order = 5,
   max_axial_order = 10,
   n_data = 1,
   #1 omnidirectional
   rendering_methods = ["omni"],
   mic_orientations = [torch.Tensor([0,0,1])],
   mic_0_gains = [{1000: 4.37,  5000: 4.37, 10000: 4.63, 15000: 5.06, 20000: 4.89}],
   #{1000: 1,  5000: 1, 10000: 1.06, 15000: 1.16, 20000: 1.12} is the standard, amplificatins measured ith SPL were introduced
   mic_180_loss=  [{1000: 0,  5000: 1, 10000: 2.75, 15000: 5, 20000: 7}]
)

###################################