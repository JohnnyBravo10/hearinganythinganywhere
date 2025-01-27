import os
import numpy as np


def compute_complement_indices(indices, n_data):
    """Given a list of indices and number of total datapoints, computes complement indices"""
    comp_indices = []
    for i in range(n_data):
        if i not in indices:
                comp_indices.append(i)

    return comp_indices


"""
Determining Training/Valid/Testing Indices for everything (Messy)
"""

class Dataset:

    """
    Class for a subdataset (e.g., classroom base dataset)   

    Constructor Parameters
    ----------------------
    load_dir: where the files for the dataset are located
    speaker_xyz: (3,) array, where speaker is in the room setup
    all_surfaces: list of Surface - surfaces definining room's geometry
    speed_of_sound: in m/s
    default_binaural_listener_forward: (3,) direction the binaural mic is facing
    default_binaural_listener_left: (3,) points left out from the binaural mic ??????????????????No, non è vero??????
    max_order: default reflection order for tracing this dataset
    max_axial_order: default reflection order for parallel walls
    """
    def __init__(self,
                load_dir,
                speaker_xyz,
                all_surfaces,
                speed_of_sound,
                default_binaural_listener_forward,
                default_binaural_listener_left,
                parallel_surface_pairs,
                train_indices,
                valid_indices,
                max_order,
                max_axial_order,
                n_data,
                ###########################################################################
                rendering_methods = None,
                mic_orientations = None,
                mic_0_gains= None, 
                mic_180_loss = None,
                cardioid_exponents = None):

        #More stuff
        self.speaker_xyz = speaker_xyz
        self.all_surfaces = all_surfaces
        self.speed_of_sound = speed_of_sound
        self.default_binaural_listener_forward = default_binaural_listener_forward
        self.default_binaural_listener_left = default_binaural_listener_left
        self.parallel_surface_pairs = parallel_surface_pairs
        self.load_dir = load_dir
        

        #indices
        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.test_indices = compute_complement_indices( list(self.train_indices)+list(self.valid_indices), n_data)

        # Default max order and axial order
        self.max_order = max_order
        self.max_axial_order = max_axial_order
        
        #####################################################################################
        
        self.rendering_methods = (rendering_methods if rendering_methods is not None else np.full(n_data, None))
        self.mic_orientations = (mic_orientations if mic_orientations is not None else np.full(n_data, None))
        self.mic_0_gains= (mic_0_gains if mic_0_gains is not None else np.full(n_data, None))
        self.mic_180_loss = (mic_180_loss if mic_180_loss is not None else np.full(n_data, None))
        self.cardioid_exponents = (cardioid_exponents if cardioid_exponents is not None else np.full(n_data, None))
        ##################################################################################

    def load_data(self):
        self.xyzs = np.load(os.path.join(self.load_dir, "xyzs.npy"))
        self.RIRs = np.load(os.path.join(self.load_dir, "RIRs.npy"), allow_pickle = True)###############allow_pickle necessario per array di oggetti (dictionaries sulle direzioni) (qui e sotto)
        self.music = np.load(os.path.join(self.load_dir, "music.npy"), allow_pickle = True)#, mmap_mode='r')################### mmap_mode = 'r' non utilizzabile su dati che sono object
        self.music_dls = np.load(os.path.join(self.load_dir, "music_dls.npy"), mmap_mode='r')
        self.bin_music_dls = np.load(os.path.join(self.load_dir, "bin_music_dls.npy"), mmap_mode='r') #!@#$
        self.bin_xyzs = np.load(os.path.join(self.load_dir, "bin_xyzs.npy"), mmap_mode='r')
        self.bin_RIRs = np.load(os.path.join(self.load_dir, "bin_RIRs.npy"), mmap_mode='r')
        self.bin_music = np.load(os.path.join(self.load_dir, "bin_music.npy"), mmap_mode='r')
        self.mic_numbers = np.load(os.path.join(self.load_dir, "mic_numbers.npy"))





all_datasets = ["classroomBase", "dampenedBase", "dampenedRotation",
 "dampenedTranslation", "dampenedPanel", "hallwayBase", "hallwayRotation", 
 "hallwayTranslation","hallwayPanel1","hallwayPanel2","hallwayPanel3",
 "complexBase","complexRotation","complexTranslation"]
 
base_datasets = ["classroomBase", "dampenedBase", "hallwayBase", "complexBase"]


def dataLoader(name):
    #Classroom Dataset
    if name[:9] == "classroom":
        import rooms.classroom as classroom
        if name=="classroomBase":
            D = classroom.BaseDataset
        #################################################
        elif name=="classroomAddedPanel":
            D = classroom.AddedPanelDataset
        #################################################
        else:
            raise ValueError('Invalid Dataset Name')

    #Dampened Room Datasets
    elif name[:8] == "dampened":
        import rooms.dampened as dampened
        if name =="dampenedBase":
            D = dampened.BaseDataset
        elif name =="dampenedRotation":
            D = dampened.RotationDataset
        elif name =="dampenedTranslation":
            D =  dampened.TranslationDataset
        elif name == "dampenedPanel":
            D = dampened.PanelDataset
        else:
            raise ValueError('Invalid Dataset Name')
    #Hallway Datasets
    elif name[:7] == "hallway":
        import rooms.hallway as hallway
        if name == "hallwayBase":
            D = hallway.BaseDataset
        elif name == "hallwayRotation":
            D = hallway.RotationDataset
        elif name == "hallwayTranslation":
            D =  hallway.TranslationDataset
        elif name == "hallwayPanel1":
            D =  hallway.PanelDataset1
        elif name == "hallwayPanel2":
            D =  hallway.PanelDataset2
        elif name == "hallwayPanel3":
            D = hallway.PanelDataset3
        else:
            raise ValueError('Invalid Dataset Name')
    elif name[:7] == "complex":
        import rooms.complex as complex
        if name == "complexBase":
            D = complex.BaseDataset
        elif name == "complexRotation":
            D = complex.RotationDataset
        elif name == "complexTranslation":
            D = complex.TranslationDataset
        else:
            raise ValueError('Invalid Dataset Name')
        
    ####################################################
    elif name[:5] == "prova":
        import rooms.prova as dataset_prova
        if name == "prova":
            D = dataset_prova.BaseDataset
    #########################################################
    elif name[:10] == "nottingham":
        if name == "nottingham_S1":
            import rooms.nottingham_S1 as nottingham_S1
            D = nottingham_S1.BaseDataset
        if name == "nottingham_S2":
            import rooms.nottingham_S2 as nottingham_S2
            D = nottingham_S2.BaseDataset
        if name == "nottingham_S3":
            import rooms.nottingham_S3 as nottingham_S3
            D = nottingham_S3.BaseDataset

    #########################################################
    elif name[:5] == "espoo":
        if name == "espoo_S2":
            import rooms.espoo_S2 as espoo_S2
            D = espoo_S2.BaseDataset
        if name == "espoo_S2_amb":
            import rooms.espoo_S2_amb as espoo_S2_amb
            D = espoo_S2_amb.BaseDataset

    else:
        raise ValueError('Invalid Dataset Name')

    D.load_data()
    return D