import numpy as np
import config
import trace1 as G
import rooms.dataset as dataset

#####################################ààà
import torch

##########################

"""
Importing this document automatically loads data from the classroom dataset
"""

speed_of_sound = 343

"""
Locations of all surfaces in Meters
"""
max_x = 7.87
max_y = 5.75
max_z = 2.91

rear_wall = G.Surface(np.array([[0, 0, 0], 
                                  [0, 0, max_z],
                                  [max_x, 0, 0]]))

front_wall = G.Surface(np.array([[0, max_y, 0],
                                  [max_x, max_y, 0],
                                  [0, max_y, max_z]]))


floor = G.Surface(np.array([[0, 0, 0],
                                [max_x, 0, 0],
                                [0, max_y, 0]]))


ceiling = G.Surface(np.array([[0, 0, max_z],
                                [0, max_y, max_z],
                               [max_x, 0, max_z]]))

left_wall = G.Surface(np.array([[0, 0, 0],
                                [0, max_y, 0],
                                 [0, 0, max_z]]))

right_wall = G.Surface(np.array([[max_x, 0, 0],[max_x, 0, max_z],                               
                                [max_x, max_y, 0]]))

walls = [rear_wall, front_wall, floor, ceiling, left_wall, right_wall]


base_surfaces = walls


"""
Train and Test Split
"""

train_indices = list([])
valid_indices = list([])


#Speaker xyz estimated from 12-point TOA, inside speaker, 8.5cm away from manual measurement.
BaseDataset = dataset.Dataset(
   load_dir = config.espoo_S1_path,
   speaker_xyz= np.array([6.37, 2.685, 1.5]), 
   all_surfaces = base_surfaces,
   speed_of_sound = speed_of_sound,
   default_binaural_listener_forward = np.array([0,1,0]),
   default_binaural_listener_left = np.array([-1,0,0]),
   parallel_surface_pairs=[[0,1], [2,3], [4,5]],
   train_indices = train_indices,
   valid_indices = valid_indices,
   max_order = 5,
   max_axial_order = 10,
   n_data = 7,
   rendering_methods = ['omni' for _ in range(7)],
   mic_orientations= [torch.Tensor([0,1,0]) for _ in range(7)],
   mic_0_gains= [{0: 0, 50:1, 1000:1, 5000: 1, 10000: 1, 15000: 1, 20000: 1} for _ in range(7)],
   mic_180_loss= [{0:0, 50:0, 1000: 0,  5000: 0, 10000: 0, 15000: 0, 20000: 0} for _ in range(7)]
)