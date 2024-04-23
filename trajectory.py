import scipy
import matplotlib.pyplot as plt
import trace
import os
import torch
import binauralize
import torchaudio.functional as F
import numpy as np
import rooms.dataset

class Trajectory:

    def __init__(self, save_dir, dataset_name, listener_xyzs, listener_forwards, listener_lefts, source_xyzs, source_axis_1s, source_axis_2s):

        self.save_dir = save_dir

        self.listener_xyzs = listener_xyzs
        self.listener_forwards = listener_forwards
        self.listener_lefts = listener_lefts
        self.source_xyzs = source_xyzs
        self.source_axis_1s = source_axis_1s
        self.source_axis_2s = source_axis_2s

        #More stuff
        self.n_points = listener_xyzs.shape[0]
        self.rirs = None
        self.D = rooms.dataset.dataLoader(dataset_name)
        self.dataset_name = dataset_name
        self.surfaces = self.D.all_surfaces
        self.speed_of_sound = self.D.speed_of_sound

        self.convolves = {}
        self.renders = {}

    def plot_traj(self, listener_xyz=None, listener_forward=None, listener_left=None, source_xyz=None, source_axis_1=None, source_axis_2=None):
    
        plt.scatter(self.listener_xyzs[:,0],self.listener_xyzs[:,1])
        plt.scatter(self.source_xyzs[:,0],self.source_xyzs[:,1])

        def plot_surface(S):
            points = np.array([S.p0, S.p1, S.opposite, S.p2, S.p0])
            plt.plot(points[:,0], points[:,1], color='black', linewidth=2)

        for surface in self.surfaces:
            plot_surface(surface)

        if listener_xyz is not None:
            plt.scatter(listener_xyz[0], listener_xyz[1])
            plt.arrow(listener_xyz[0],listener_xyz[1],listener_forward[0],listener_forward[1],color='red',head_width=0.1,label="Listener Direction")
            plt.arrow(listener_xyz[0],listener_xyz[1],listener_left[0],listener_left[1],color='green', head_width=0.1)

        if source_xyz is not None:
            plt.scatter(source_xyz[0], source_xyz[1])
            plt.arrow(source_xyz[0],source_xyz[1],source_axis_1[0],source_axis_1[1],color='orange',head_width=0.1,label="Source Direction")
            plt.arrow(source_xyz[0],source_xyz[1],source_axis_2[0],source_axis_2[1],color='pink', head_width=0.1)
            
        plt.axis('scaled')
        plt.legend()
    
    def gif(self):

        directory = os.path.join(self.save_dir,"images/")
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i in range(self.n_points-1):
            
            source_axis_1_rotated = np.array([self.source_axis_1s[i][1], -self.source_axis_1s[i][0], self.source_axis_1s[i][2]])
            source_axis_2_rotated = np.array([self.source_axis_2s[i][1], -self.source_axis_2s[i][0], self.source_axis_2s[i][2]])

            self.plot_traj(listener_xyz=self.listener_xyzs[i],
                           listener_forward=self.listener_forwards[i],
                           listener_left=self.listener_lefts[i],
                           source_xyz=self.source_xyzs[i],
                           source_axis_1=source_axis_1_rotated,
                           source_axis_2=source_axis_2_rotated)
            plt.savefig(os.path.join(directory, str(i).zfill(6)+".png"))
            plt.close()

    def save_paths(self):

        for subdir in ["reflections", "transmissions", "delays", "starts", "ends"]:
            directory = os.path.join(self.save_dir,subdir)
            if not os.path.exists(directory):
                os.makedirs(directory)

        if self.dataset_name[:8] == "dampened":
            max_order = 5
            max_axial_order = 6
        elif self.dataset_name[:9] == "classroom":
            max_order = 5 !@#$
            max_axial_order = 10    
        elif self.dataset_name[:7] == "complex":
            max_order = 4
            max_axial_order = 10
        elif self.dataset_name[:7] == "hallway":
            max_order = 5
            max_axial_order = 50
        else:
            raise ValueError

        for i in range(self.listener_xyzs.shape[0]):
            dest = self.listener_xyzs[i]
            source = self.source_xyzs[i]


            reflection_path_indices, transmission_path_indices, delays, start_directions, end_directions = trace.get_reflections_transmissions_and_delays(source=source, dest=dest, surfaces=self.surfaces, speed_of_sound=self.speed_of_sound, max_order=max_order, candidate_transmission_surface_indices=None, img_save_dir=None, parallel_surface_pairs=self.D.parallel_surface_pairs, max_axial_order=max_axial_order)

            np.save(os.path.join(self.save_dir,"reflections/"+str(i)+".npy"), np.array(reflection_path_indices, dtype=object))
            np.save(os.path.join(self.save_dir,"transmissions/"+str(i)+".npy"), np.array(transmission_path_indices, dtype=object))
            np.save(os.path.join(self.save_dir,"delays/"+str(i)+".npy"), delays)
            np.save(os.path.join(self.save_dir,"starts/"+str(i)+".npy"), start_directions)
            np.save(os.path.join(self.save_dir,"ends/"+str(i)+".npy"), end_directions)
            print(i)
    
    def get_RIRs(self, R, model_path, reflections=True):

        if model_path is not None:
            R.load_state_dict(torch.load(model_path)['model_state_dict'],strict=False)
        directory = os.path.join(self.save_dir,"rirs")
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        for idx in range(self.n_points-1):
            print(idx)
            listener_xyz = self.listener_xyzs[idx]
            listener_forward = self.listener_forwards[idx]
            listener_left = self.listener_lefts[idx]

            source_xyz = self.source_xyzs[idx]
            source_axis_1 = self.source_axis_1s[idx]
            source_axis_2 = self.source_axis_2s[idx]
            
            
            rir = binauralize.render_binaural(R=R, source_xyz=source_xyz, source_axis_1=source_axis_1, source_axis_2=source_axis_2, listener_xyz=listener_xyz, listener_forward=listener_forward, listener_left=listener_left, surfaces=self.surfaces, speed_of_sound=self.speed_of_sound, RIR_only=True, load_dir=self.save_dir, load_num=idx,reflections=reflections, plot=False)

            np.save(os.path.join(self.save_dir, "rirs/" + str(idx) + ".npy"), rir)
        
        print("rendering done")

        
    def load_RIRs(self):
        rirs = []
        for idx in range(self.n_points-1):
            rirs.append(np.load(os.path.join(self.save_dir, "rirs/" + str(idx) + ".npy")))
        rirs = np.array(rirs)
        np.save(os.path.join(self.save_dir, "rirs.npy"), rirs)
        self.rirs = rirs
        return rirs

    def convolve_audio(self, source_name, root_dir="/viscam/projects/audio_nerf/masonstuff/trajectories/musicfiles",full=True, medium=False, cutoff=None):    

        if self.rirs is None:
            print("Error: RIRs not initialized")
            return
        
        print('bruh')

        fs, source = scipy.io.wavfile.read(os.path.join(root_dir, source_name))
        #!@#$
        if not full:
            source = source[:1440000]
        
        if medium:
            source = source[:1897920]
        
        if cutoff is not None:
            source = source[:cutoff]

        assert fs==48000
        
        if source.ndim > 1:
            source = source[:,0]
        
        print(source.shape)
            
        total_length =  source.shape[-1] - source.shape[-1]%(self.n_points-1)
        print(total_length)
        source = torch.Tensor(source.reshape(1,-1)).cuda()
        rirs = torch.Tensor(self.rirs).cuda()

        convolved = torch.zeros(self.n_points-1, 2, total_length)
        
        for i in range(self.rirs.shape[0]):
            convolved[i] = F.fftconvolve(rirs[i], source)[...,:total_length]

        convolved = convolved.cpu().numpy()
        self.convolves[source_name] = convolved

        return convolved
    
    def fill(self, source_name, fade_length=20000):
        
        convolved = self.convolves[source_name]
        total_length = convolved.shape[-1]

        render = np.zeros((2, total_length))
        window = scipy.signal.get_window("bartlett", fade_length*2)
        fade_in = window[:fade_length]
        fade_out = window[fade_length:]

        hop_length = total_length/(self.n_points-1)
        print(hop_length)
        hop_length = int(hop_length)

        for i in range(self.n_points-1):
            if i==0:
                start_clip = 0
                end_clip = hop_length
                render[:,start_clip:end_clip] += convolved[i, :, start_clip:end_clip]
                render[:,end_clip:end_clip+fade_length] +=  convolved[i, :, end_clip:end_clip+fade_length]*fade_out
                
            else:
                start_clip = hop_length*i
                end_clip = hop_length*(i+1)
                render[:,start_clip-fade_length:start_clip] +=  convolved[i,:,  start_clip-fade_length:start_clip]*fade_in
                render[:,start_clip:end_clip] +=  convolved[i, :, start_clip:end_clip]
                if i != self.n_points - 2:
                    render[:,end_clip:end_clip+fade_length] +=  convolved[i, :,  end_clip:end_clip+fade_length]*fade_out            
        

        render_normalized = (render/np.max(render))*32767
        self.renders[source_name] = (render/np.max(render))
        scipy.io.wavfile.write(os.path.join(self.save_dir,source_name), 48000, (render_normalized.T).astype(np.int16))
        
        return render