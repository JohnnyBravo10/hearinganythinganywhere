import numpy as np
import config
import trace1 as G
import rooms.dataset as dataset

import torch ##############################


"""
Importing this document automatically loads data from the dampened room dataset
"""

speed_of_sound = 343

"""
Locations of all surfaces
"""

cm = 0.01
max_x = 485*cm
max_y = 519.5*cm
max_z = 273.1*cm

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


parallel_surface_pairs = [[2,3]]


# Locations of Panel
panel_y = (225.5-1.3)*cm
panel_x1 = 191*cm
panel_x2 = 287.5*cm
panel_z1 = 14.5*cm
panel_z2 = 185.5*cm

panel = G.Surface(np.array([[panel_x1, panel_y, panel_z2],[panel_x1, panel_y, panel_z1],                               
                                [panel_x2, panel_y, panel_z2]]))



"""
Train, Test indices, Making Dataset class instances
"""

train_indices_base = [0, 23, 46, 69, 92+12, 115, 138, 161, 184, 207, 230, 253]
valid_indices_base = dataset.compute_complement_indices(train_indices_base + list(np.arange(138)*2), 276)[::2]

# Default tracing orders
max_order = 5
max_axial_order = 6

#speaker_xyz = np.array([2.4051, 2.5638, 1.2334]), #Ground Truth, estimated 10/31 using 276 points, TOA. Error = 4 cm.
BaseDataset = dataset.Dataset(
    load_dir = config.dampenedBase_path,
    speaker_xyz = np.array([2.4542, 2.4981, 1.2654]), #Estimated 11/14, error 11 cm
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs,
    train_indices = train_indices_base,
    valid_indices = valid_indices_base,
    max_order = max_order,
    max_axial_order = max_axial_order,
    n_data = 276,

    ####################################
    rendering_methods =["omni" for _ in range(276)],
    mic_orientations = [torch.Tensor([-1,0,0]) for _ in range(276)],


    mic_0_gains = np.tile([{500: 1.02, 1000: 1, 5000: 0.90, 10000: 1.23, 20000: 1.35}, 
                   {500: 1.11, 1000: 1, 5000: 0.79, 10000: 0.93, 20000: 1.48}, 
                   {500: 0.89, 1000: 1, 5000: 0.82, 10000: 1.11, 20000:1.32}, 
                   {500: 0.99, 1000: 1, 5000: 0.84, 10000: 1.07, 20000: 1.36},
                   {500: 1.16, 1000: 1, 5000: 0.83, 10000: 1.11, 20000:1.45},
                   {500: 0.93, 1000: 1, 5000: 0.91, 10000: 1.26, 20000:1.58},
                   {500: 0.92, 1000: 1, 5000: 0.93, 10000: 1.66, 20000:2.29},
                   {500: 1.12, 1000: 1, 5000: 0.88, 10000: 1.27, 20000:1.51},
                   {500: 1.01, 1000: 1, 5000: 0.91, 10000: 1.26, 20000:1.55},
                   {500: 1.01, 1000: 1, 5000: 0.95, 10000: 1.12, 20000:1.64},
                   {500: 1.11, 1000: 1, 5000: 0.84, 10000: 1.04, 20000:1.27},
                   {500: 1.20, 1000: 1, 5000: 0.97, 10000: 1.35, 20000:1.70}], 23), 
    mic_180_loss=  [{500: 0, 1000: 0.3, 5000: 1.60, 10000: 3.75, 20000:7} for _ in range(276)]

    ######################################
)

BaseDataset_no_mic_char = dataset.Dataset(
    load_dir = config.dampenedBase_path,
    speaker_xyz = np.array([2.4542, 2.4981, 1.2654]), #Estimated 11/14, error 11 cm
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs,
    train_indices = train_indices_base,
    valid_indices = valid_indices_base,
    max_order = max_order,
    max_axial_order = max_axial_order,
    n_data = 276,
    
    rendering_methods =[None for _ in range(276)], 
    mic_orientations = [None for _ in range(276)], 
    mic_0_gains = [None for _ in range(276)], 
    mic_180_loss=  [None for _ in range(276)])


train_indices_120 = np.append((np.arange(11)*11), 109)
valid_indices_120 = dataset.compute_complement_indices(train_indices_120, 120)[::2]

#speaker_xyz = np.array([2.4612, 2.6852, 1.2366]),  #Estimated 10/31 using 120 points, TOA. Error = 11 cm.
RotationDataset = dataset.Dataset(
    load_dir = config.dampenedRotation_path,
    speaker_xyz=np.array([2.4595, 2.6748, 1.0659]), #Estimated 11/14, error = 17 cm from manual measurement
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs,
    train_indices = train_indices_120,
    valid_indices = valid_indices_120,
    max_order = max_order,
    max_axial_order = max_axial_order,
    n_data = 120
    
)




#speaker_xyz = np.array([1.12, 0.79, 1.2366]),  #Measured, Z borrowed from above
TranslationDataset = dataset.Dataset(
    load_dir = config.dampenedTranslation_path,
    speaker_xyz=np.array([1.2621, 0.5605, 1.2404]), #error 27 cm from manual measurement, 11/14
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs,
    train_indices = train_indices_120,
    valid_indices = valid_indices_120,
    max_order = max_order,
    max_axial_order = max_axial_order,
    n_data = 120
)



train_indices_panel = np.array([0,  23,  46,  69,  92+12, 115, 138, 161, 184, 207, 230, 241])
valid_indices_panel = dataset.compute_complement_indices(list(train_indices_panel) + list(np.arange(126)*2), 252)[::2]

#speaker_xyz = np.array([2.4051, 2.5638, 1.2334])
PanelDataset = dataset.Dataset(
    load_dir = config.dampenedPanel_path,
    speaker_xyz = np.array([2.4052, 2.5292, 1.3726]), # error = 18 cm from manual measurement, 11/4
    all_surfaces = walls+[panel],
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs,
    train_indices = train_indices_panel,
    valid_indices = valid_indices_panel,
    max_order = max_order,
    max_axial_order = max_axial_order,
    n_data = 252
)