import os
import math
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torchaudio.functional as F
import trace1

#############################
from scipy.special import sph_harm

import torch.nn.functional as Func


import scipy.fft
##############################


torch.set_default_dtype(torch.float32)

class Renderer(nn.Module):
    """
    Class for a RIR renderer.

    Constructor Parameters
    ----------
    n_surfaces: int
        number of surfaces to model in the room
    RIR_length: int
        length of the RIR in samples
    filter_length: int
        length of each reflection's contribution to the RIR, in samples
    source_response_length: int
        length of the convolutional kernel used to model the sound source
    surface_freqs: list of int
        frequencies in Hz to fit each surface's reflection response at, the rest are interpolated
    dir_freqs: list of int
        frequencies in Hz to fit the source's directivity response at, the rest are interpolated
    n_fibonacci: int
        number of points to distribute on the unit sphere,
        at which where the speaker's directivity is fit.
    spline_indices: list of int
        times (in samples) at which the late/early stage spline is fit
    toa_perturb: bool
        if times of arrival are perturbed (used during training)
    model_transmission: bool
        if we are modeling surface transmission as well.
    """
    def __init__(self, 
                n_surfaces,
                RIR_length=96000, filter_length=1023, source_response_length=1023,
                surface_freqs=[32, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
                dir_freqs = [32, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
                n_fibonacci = 128, sharpness = 8,
                late_stage_model = "UniformResidual",
                spline_indices = [200, 500, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
                                  6000, 7000, 8000, 10000, 12000, 14000, 16000, 18000, 20000,
                                  22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000,
                                  38000, 40000, 44000, 48000, 56000, 70000, 80000],
                toa_perturb=True,
                model_transmission=False,
                fs=48000):

        super().__init__()

        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nyq = fs/2

        # Arguments
        self.n_surfaces = n_surfaces
        self.RIR_length = RIR_length
        self.filter_length = filter_length
        self.source_response_length = source_response_length
        self.surface_freqs = torch.tensor(surface_freqs)
        self.dir_freqs = torch.tensor(dir_freqs)
        self.n_fibonacci = n_fibonacci
        self.sharpness = sharpness
        self.late_stage_model = late_stage_model
        self.spline_indices = spline_indices
        self.toa_perturb = toa_perturb
        self.model_transmission = model_transmission

        # Other Attributes
        self.n_surface_freqs = len(surface_freqs)
        self.n_dir_freqs = len(dir_freqs)
        self.samples = torch.arange(self.RIR_length).to(self.device)
        self.times = self.samples/fs
        self.sigmoid = nn.Sigmoid()

        # Early and Late reflection initialization
        self.init_early()
        self.init_late()

        # Spline
        self.n_spline = len(spline_indices)
        self.IK = get_time_interpolator(n_target=RIR_length, indices=torch.tensor(spline_indices)).to(self.device)   
        self.spline_values = nn.Parameter(torch.linspace(-5, 5, self.n_spline))

        ##########################################################################################
        # Beampattern orders cutoff frequencies (for the moment this parameter is detached during learning, not learned)
        self.bp_ord_cut_freqs = nn.Parameter(torch.Tensor([70, 400, 800, 1000, 1300, 2000]))
        ##########################################################################################

    def init_early(self):
        
        ##################### A could be initialized in a different way exploiting a priori knowledge on the surfaces materials 
         
        # Initializing "Energy Vector", which stores the coefficients for each surface
        if self.model_transmission:
            # Second axis indices are specular reflection, transmission, and absorption
            A = torch.zeros(self.n_surfaces,3, self.n_surface_freqs)
        else:
            # Second axis indices are specular reflection and absorption
            A = torch.zeros(self.n_surfaces,2,self.n_surface_freqs)
        
        self.energy_vector = nn.Parameter(A)

        # Setting up Frequency responses
        n_freq_samples = 1 + 2 ** int(math.ceil(math.log(self.filter_length, 2))) # is there any particular reason to choose filter_length?
        
        self.freq_grid = torch.linspace(0.0, self.nyq, n_freq_samples)
        
        surface_freq_indices = torch.round(self.surface_freqs*((n_freq_samples-1)/self.nyq)).int() 
        

        self.surface_freq_interpolator = get_interpolator(n_freq_target=n_freq_samples, 
                                                          freq_indices=surface_freq_indices).to(self.device)

        dir_freq_indices = torch.round(self.dir_freqs*((n_freq_samples-1)/self.nyq)).int()
        self.dir_freq_interpolator = get_interpolator(n_freq_target=n_freq_samples,
                                                      freq_indices=dir_freq_indices).to(self.device)
        
        

        self.window = torch.Tensor(
            scipy.fft.fftshift(scipy.signal.get_window("hamming", self.filter_length, fftbins=False))).to(self.device)

        # Source Response
        source_response = torch.zeros(self.source_response_length)
        source_response[0] = 0.1 # Initialize to identity
        self.source_response = nn.Parameter(source_response)

        # Directivity Pattern
        self.sphere_points = torch.tensor(fibonacci_sphere(self.n_fibonacci)).to(self.device)
        self.directivity_sphere = nn.Parameter(torch.ones(self.n_fibonacci, self.n_dir_freqs))

        # Parameter for energy decay over time
        self.decay = nn.Parameter(5*torch.ones(1))


    def init_late(self):
        """Initializing the late-stage model - other methods besides a uniform residual may be explored in the future"""
        if self.late_stage_model == "UniformResidual":
            self.RIR_residual = nn.Parameter(torch.zeros(self.RIR_length))
        else:
            raise ValueError("Invalid Residual Mode")

    def render_early(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        n_paths = loc.delays.shape[0]
        device = self.device

        # Energy coefficients: softmax and amplitude computation
        energy_coeffs = nn.functional.softmax(self.energy_vector, dim=-2)
        amplitudes = torch.sqrt(energy_coeffs).to(device)

        # Mask and gains_profile computation
        mask = torch.stack((loc.reflection_mask, loc.transmission_mask), dim=-1).unsqueeze(-1).to(device)
        gains_profile = (amplitudes[:, :2, :].unsqueeze(0) ** mask).unsqueeze(-1)
        
        # Compute reflection frequency response in a single step
        reflection_response = self.surface_freq_interpolator * gains_profile
        reflection_frequency_response = torch.prod(torch.sum(reflection_response, dim=-2), dim=-3).prod(dim=-2)
        
        # Handle source rotation if present
        start_dirs = loc.start_directions_normalized.to(device)
        if source_axis_1 is not None and source_axis_2 is not None:
            source_basis = torch.tensor(np.stack((source_axis_1, source_axis_2, np.cross(source_axis_1, source_axis_2)), axis=-1), 
                                        dtype=torch.double, device=device)
            start_dirs = start_dirs @ source_basis
        
        # Directivity response calculation optimized
        dots = start_dirs @ self.sphere_points.T
        weights = torch.exp(-self.sharpness * (1 - dots))
        weights /= weights.sum(dim=-1, keepdim=True)
        directivity_profile = (weights.unsqueeze(-1) * self.directivity_sphere).sum(dim=1)
        directivity_response = torch.sum(directivity_profile.unsqueeze(-1) * self.dir_freq_interpolator, dim=-2)
        freq_response = torch.exp(directivity_response) * reflection_frequency_response
        
        # Hilbert transform and inverse FFT
        phases = hilbert_one_sided(safe_log(freq_response), device=device)
        
        
        out_full = torch.fft.irfft(freq_response * torch.exp(1j * phases))
        
        out = out_full[..., :self.filter_length] * self.window

        # Reflection kernels calculation without loop
        delays = (loc.delays.to(self.device) + (torch.round(7 * torch.randn(n_paths, device=self.device)) if self.toa_perturb else 0)).long()

        reflection_kernels = torch.zeros((n_paths, self.RIR_length), device=device)
        # Converti reflection_kernels a float64 se necessario, o viceversa
        reflection_kernels = torch.zeros((n_paths, self.RIR_length), device=self.device, dtype=torch.float64)

        # Assicurati che anche l'operazione con 'out' sia coerente
        out = out.to(torch.float64)  # Converti 'out' a float64

        # Poi esegui scatter_add_
        reflection_kernels.scatter_add_(
            1, 
            (delays[:, None] + torch.arange(out.shape[-1], device=self.device)[None, :]).long(),
            (out * (2 * self.nyq / 343) / delays[:, None]).to(torch.float64)
        )

        # Transmission mask application
        if not self.model_transmission:
            reflection_kernels *= (loc.transmission_mask.sum(dim=-1) == 0).unsqueeze(1).to(device)

        # Convolution with HRIRs (if provided)
        RIR_early = torch.sum(reflection_kernels, dim=0)
        if hrirs is not None:
            RIR_early = F.fftconvolve(reflection_kernels.unsqueeze(1), hrirs.to(device)).sum(axis=0)
        
        RIR_early = F.fftconvolve(self.source_response - self.source_response.mean(), RIR_early)[:self.RIR_length]
        RIR_early *= (self.sigmoid(self.decay) ** self.times)

        return RIR_early
    
    
    #######################################################################
    def render_early_music_instrument(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        """
        Renders the early-stage RIR without considering the source impulse response

        Parameters
        ----------
        loc: ListenerLocation
            characterizes the location at which we render the early-stage RIR
        hrirs: np.array (n_paths x 2 x h_rir_length)
            head related IRs for each reflection path's direction
        source_axis_1: np.array (3,)
            first axis specifying virtual source rotation,
            default is None which is (1,0,0)
        source_axis_2: np.array (3,)
            second axis specifying virtual source rotation,
            default is None which is (0,1,0)        

        Returns
        -------
        RIR_early - (N,) tensor, early-stage RIR
        """

        """
        Computing Reflection Response
        """
        n_paths = loc.delays.shape[0]
        device = self.device

        # Energy coefficients: softmax and amplitude computation
        energy_coeffs = nn.functional.softmax(self.energy_vector, dim=-2)
        amplitudes = torch.sqrt(energy_coeffs).to(device)

        # Mask and gains_profile computation
        mask = torch.stack((loc.reflection_mask, loc.transmission_mask), dim=-1).unsqueeze(-1).to(device)
        gains_profile = (amplitudes[:, :2, :].unsqueeze(0) ** mask).unsqueeze(-1)
        
        # Compute reflection frequency response in a single step
        reflection_response = self.surface_freq_interpolator * gains_profile
        reflection_frequency_response = torch.prod(torch.sum(reflection_response, dim=-2), dim=-3).prod(dim=-2)
        
        # Handle source rotation if present
        start_dirs = loc.start_directions_normalized.to(device)
        if source_axis_1 is not None and source_axis_2 is not None:
            source_basis = torch.tensor(np.stack((source_axis_1, source_axis_2, np.cross(source_axis_1, source_axis_2)), axis=-1), 
                                        dtype=torch.double, device=device)
            start_dirs = start_dirs @ source_basis
        
        # Directivity response calculation optimized
        dots = start_dirs @ self.sphere_points.T
        weights = torch.exp(-self.sharpness * (1 - dots))
        weights /= weights.sum(dim=-1, keepdim=True)
        directivity_profile = (weights.unsqueeze(-1) * self.directivity_sphere).sum(dim=1)
        directivity_response = torch.sum(directivity_profile.unsqueeze(-1) * self.dir_freq_interpolator, dim=-2)
        freq_response = torch.exp(directivity_response) * reflection_frequency_response
        
        # Hilbert transform and inverse FFT
        phases = hilbert_one_sided(safe_log(freq_response), device=device)
        out_full = torch.fft.irfft(freq_response * torch.exp(1j * phases))
        out = out_full[..., :self.filter_length] * self.window

        # Reflection kernels calculation without loop
        delays = (loc.delays.to(self.device) + (torch.round(7 * torch.randn(n_paths, device=self.device)) if self.toa_perturb else 0).to(self.device)).long()

        reflection_kernels = torch.zeros((n_paths, self.RIR_length), device=device)
        # Converti reflection_kernels a float64 se necessario, o viceversa
        reflection_kernels = torch.zeros((n_paths, self.RIR_length), device=self.device, dtype=torch.float64)

        # Assicurati che anche l'operazione con 'out' sia coerente
        out = out.to(torch.float64)  # Converti 'out' a float64

        # Poi esegui scatter_add_
        reflection_kernels.scatter_add_(
            1, 
            (delays[:, None] + torch.arange(out.shape[-1], device=self.device)[None, :]).long(),
            (out * (2 * self.nyq / 343) / delays[:, None]).to(torch.float64)
        )

        # Transmission mask application
        if not self.model_transmission:
            reflection_kernels *= (loc.transmission_mask.sum(dim=-1) == 0).unsqueeze(1).to(device)

        # Convolution with HRIRs (if provided)
        RIR_early = torch.sum(reflection_kernels, dim=0)
        if hrirs is not None:
            RIR_early = F.fftconvolve(reflection_kernels.unsqueeze(1), hrirs.to(device)).sum(axis=0)
        
        
        RIR_early *= (self.sigmoid(self.decay) ** self.times)

        return RIR_early
    
    
    #####################################################################
    def render_early_microphone_response(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        """
        Optimized version of the early-stage RIR rendering method.  
        """

        n_paths = loc.delays.shape[0]
        device = self.device

        # Energy coefficients: softmax and amplitude computation
        energy_coeffs = nn.functional.softmax(self.energy_vector, dim=-2)
        amplitudes = torch.sqrt(energy_coeffs).to(device)

        # Mask and gains_profile computation
        mask = torch.stack((loc.reflection_mask, loc.transmission_mask), dim=-1).unsqueeze(-1).to(device)
        gains_profile = (amplitudes[:, :2, :].unsqueeze(0) ** mask).unsqueeze(-1)
        
        # Compute reflection frequency response in a single step
        reflection_response = self.surface_freq_interpolator * gains_profile
        reflection_frequency_response = torch.prod(torch.sum(reflection_response, dim=-2), dim=-3).prod(dim=-2)
        
        # Handle source rotation if present
        start_dirs = loc.start_directions_normalized.to(device)
        if source_axis_1 is not None and source_axis_2 is not None:
            source_basis = torch.tensor(np.stack((source_axis_1, source_axis_2, np.cross(source_axis_1, source_axis_2)), axis=-1), 
                                        dtype=torch.double, device=device)
            start_dirs = start_dirs @ source_basis
        
        # Directivity response calculation optimized
        dots = start_dirs @ self.sphere_points.T
        weights = torch.exp(-self.sharpness * (1 - dots))
        weights /= weights.sum(dim=-1, keepdim=True)
        directivity_profile = (weights.unsqueeze(-1) * self.directivity_sphere).sum(dim=1)
        directivity_response = torch.sum(directivity_profile.unsqueeze(-1) * self.dir_freq_interpolator, dim=-2)
        freq_response = torch.exp(directivity_response) * reflection_frequency_response

        # Microphone response calculation using broadcasting instead of loops
        mic_freqs = torch.tensor(list(self.mic_180_loss.keys()), device=self.device)
        mic_indices = torch.round(mic_freqs * (freq_response.shape[1] / self.nyq)).long().to(self.device)
        self.mic_freq_interpolator = get_interpolator(freq_response.shape[1], mic_indices)
        
        # Vectorized angle computation and gain application
        angles = torch.acos((torch.matmul(loc.end_directions_normalized.double(), -self.mic_direction.double())))

        mic_loss_factors = torch.stack([torch.pow(10, -(angles * self.mic_180_loss[f.item()] / math.pi) / 20) * self.mic_0_gain[f.item()] for f in mic_freqs], dim=-1)
        
        mic_response = torch.sum(mic_loss_factors.unsqueeze(-1) * self.mic_freq_interpolator, dim=-2)

        freq_response *= mic_response.to(self.device)
        
       # Hilbert transform and inverse FFT
        phases = hilbert_one_sided(safe_log(freq_response), device=self.device)
        
        
        out_full = torch.fft.irfft(freq_response * torch.exp(1j * phases))
        
        out = out_full[..., :self.filter_length] * self.window

        # Reflection kernels calculation without loop
        delays = (loc.delays.to(self.device) + (torch.round(7 * torch.randn(n_paths, device=self.device)) if self.toa_perturb else 0)).long()

        reflection_kernels = torch.zeros((n_paths, self.RIR_length), device=self.device)
        # Converti reflection_kernels a float64 se necessario, o viceversa
        reflection_kernels = torch.zeros((n_paths, self.RIR_length), device=self.device, dtype=torch.float64)

        # Assicurati che anche l'operazione con 'out' sia coerente
        out = out.to(torch.float64)  # Converti 'out' a float64

        # Poi esegui scatter_add_
        reflection_kernels.scatter_add_(
            1, 
            (delays[:, None] + torch.arange(out.shape[-1], device=self.device)[None, :]).long(),
            (out * (2 * self.nyq / 343) / delays[:, None]).to(torch.float64)
        )

        # Transmission mask application
        if not self.model_transmission:
            reflection_kernels *= (loc.transmission_mask.sum(dim=-1) == 0).unsqueeze(1).to(self.device)

        # Convolution with HRIRs (if provided)
        RIR_early = torch.sum(reflection_kernels, dim=0)
        if hrirs is not None:
            RIR_early = F.fftconvolve(reflection_kernels.unsqueeze(1), hrirs.to(self.device)).sum(axis=0)
        
        RIR_early = F.fftconvolve(self.source_response - self.source_response.mean(), RIR_early)[:self.RIR_length]
        RIR_early *= (self.sigmoid(self.decay) ** self.times)

        return RIR_early

    
    #####################################################################
    def render_early_cardioid(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        """
        Optimized version of early-stage RIR rendering with cardioid microphone response.
        """
        n_paths = loc.delays.shape[0]
        device = self.device

        # Energy coefficients: softmax and amplitude computation
        energy_coeffs = nn.functional.softmax(self.energy_vector, dim=-2)
        amplitudes = torch.sqrt(energy_coeffs).to(device)

        # Mask and gains_profile computation
        mask = torch.stack((loc.reflection_mask, loc.transmission_mask), dim=-1).unsqueeze(-1).to(device)
        gains_profile = (amplitudes[:, :2, :].unsqueeze(0) ** mask).unsqueeze(-1)
        
        # Compute reflection frequency response in a single step
        reflection_response = self.surface_freq_interpolator * gains_profile
        reflection_frequency_response = torch.prod(torch.sum(reflection_response, dim=-2), dim=-3).prod(dim=-2)
        
        # Handle source rotation if present
        start_dirs = loc.start_directions_normalized.to(device)
        if source_axis_1 is not None and source_axis_2 is not None:
            source_basis = torch.tensor(np.stack((source_axis_1, source_axis_2, np.cross(source_axis_1, source_axis_2)), axis=-1), 
                                        dtype=torch.double, device=device)
            start_dirs = start_dirs @ source_basis
        
        # Directivity response calculation optimized
        dots = start_dirs @ self.sphere_points.T
        weights = torch.exp(-self.sharpness * (1 - dots))
        weights /= weights.sum(dim=-1, keepdim=True)
        directivity_profile = (weights.unsqueeze(-1) * self.directivity_sphere).sum(dim=1)
        directivity_response = torch.sum(directivity_profile.unsqueeze(-1) * self.dir_freq_interpolator, dim=-2)
        freq_response = torch.exp(directivity_response) * reflection_frequency_response

        # Efficient microphone response calculation
        mic_freqs = torch.tensor(list(self.cardioid_exponents.keys()), device=self.device)
        mic_indices = torch.round(mic_freqs * (freq_response.shape[1] / self.nyq)).long()
        mic_interpolator = get_interpolator(freq_response.shape[1], mic_indices).to(self.device)
        
        # Vectorized cosine and cardioid calculation
        cosines = torch.matmul(loc.end_directions_normalized.double(), -self.mic_direction.double()).to(self.device)
        card_amp = ((1 + cosines[:, None]) / 2) ** torch.tensor(
            list(self.cardioid_exponents.values()), device=self.device
        )
        decibel_loss = (1 - card_amp) * 25
        mic_resp_profile = torch.pow(10, -decibel_loss / 20) * torch.tensor(
            list(self.mic_0_gain.values()), device=self.device
        )
        
        mic_response = torch.sum(mic_resp_profile.unsqueeze(-1) * mic_interpolator, dim=-2)
        
        freq_response *= mic_response
        
        # Hilbert transform and inverse FFT
        phases = hilbert_one_sided(safe_log(freq_response), device=device)
        
        
        out_full = torch.fft.irfft(freq_response * torch.exp(1j * phases))
        
        out = out_full[..., :self.filter_length] * self.window

        # Reflection kernels calculation without loop
        delays = (loc.delays.to(self.device) + (torch.round(7 * torch.randn(n_paths, device=self.device)) if self.toa_perturb else 0)).long()

        reflection_kernels = torch.zeros((n_paths, self.RIR_length), device=device)
        # Converti reflection_kernels a float64 se necessario, o viceversa
        reflection_kernels = torch.zeros((n_paths, self.RIR_length), device=self.device, dtype=torch.float64)

        # Assicurati che anche l'operazione con 'out' sia coerente
        out = out.to(torch.float64)  # Converti 'out' a float64

        # Poi esegui scatter_add_
        reflection_kernels.scatter_add_(
            1, 
            (delays[:, None] + torch.arange(out.shape[-1], device=self.device)[None, :]).long(),
            (out * (2 * self.nyq / 343) / delays[:, None]).to(torch.float64)
        )

        # Transmission mask application
        if not self.model_transmission:
            reflection_kernels *= (loc.transmission_mask.sum(dim=-1) == 0).unsqueeze(1).to(device)

        # Convolution with HRIRs (if provided)
        RIR_early = torch.sum(reflection_kernels, dim=0)
        if hrirs is not None:
            RIR_early = F.fftconvolve(reflection_kernels.unsqueeze(1), hrirs.to(device)).sum(axis=0)
        
        RIR_early = F.fftconvolve(self.source_response - self.source_response.mean(), RIR_early)[:self.RIR_length]
        RIR_early *= (self.sigmoid(self.decay) ** self.times)

        return RIR_early
    
    
    ##############################################################################
     


    ##############################################################################
    # Doesn't support hrirs
    def render_early_directional(self, loc, azimuths, elevations, source_axis_1=None, source_axis_2=None, listener_forward = np.array([0,1,0]), listener_left = np.array([-1,0,0])):
        """
        Renders the early-stage RIR

        Parameters
        ----------
        loc: ListenerLocation
            characterizes the location at which we render the early-stage RIR
        azimuths: list of floats (2 decimal signs, [-180,180]) 
            azimuths of the incoming direction to predict
        elevations: list of floats (2 decimal signs, [-90,90]) 
            elevations of the incoming direction to predict
        source_axis_1: np.array (3,)
            first axis specifying virtual source rotation,
            default is None which is (1,0,0)
        source_axis_2: np.array (3,)
            second axis specifying virtual source rotation,
            default is None which is (0,1,0)
        angular_sensitivity: float
            angle that determines the number of directions considered      

        Returns
        -------
        directional_freq_responses - list of dictionaries 
             for each direction, a time domain (t_response) and a frequency domain(f_response) early RIR are specified
        """

        """
        Computing Reflection Response
        """

        n_paths = loc.delays.shape[0]
        energy_coeffs = nn.functional.softmax(self.energy_vector, dim=-2) # Conservation of energy
        amplitudes = torch.sqrt(energy_coeffs).to(self.device)

        # mask is n_paths * n_surfaces * 2 * 1 - 1 at (path, surface, 0) indicates
        # path reflects off surface
        mask = torch.stack((loc.reflection_mask, loc.transmission_mask), dim=-1).unsqueeze(-1).to(self.device)
        
        # gains_profile is n_paths * n_surfaces * 2 * num_frequencies * 1
        if not self.model_transmission:  
            paths_without_transmissions = torch.sum(loc.transmission_mask, dim=-1) == 0
        gains_profile = (amplitudes[:,0:2,:].unsqueeze(0)**mask).unsqueeze(-1)

        # reflection_frequency_response = n_paths * n_freq_samples
        reflection_frequency_response = torch.prod(torch.prod(
            torch.sum(self.surface_freq_interpolator*gains_profile, dim=-2),dim=-3),dim=-2)##########not clear??


        """
        Computing Directivity Response
        """
        start_directions_normalized = loc.start_directions_normalized.to(self.device)

        # If there is speaker rotation
        if source_axis_1 is not None and source_axis_2 is not None:
            source_axis_3 = np.cross(source_axis_1 ,source_axis_2)
            source_basis = np.stack( (source_axis_1, source_axis_2, source_axis_3), axis=-1)
            start_directions_normalized_transformed = (
                start_directions_normalized @ torch.Tensor(source_basis).double().cuda())
            dots =  start_directions_normalized_transformed @ (self.sphere_points).T
        else:
            dots = start_directions_normalized @ (self.sphere_points).T

        
        # Normalized weights for each directivity bin
        weights = torch.exp(-self.sharpness*(1-dots))
        weights = weights/(torch.sum(weights, dim=-1).view(-1, 1))
        weighted = weights.unsqueeze(-1)* self.directivity_sphere
        directivity_profile = torch.sum(weighted, dim=1)
        directivity_response = torch.sum(directivity_profile.unsqueeze(-1) * self.dir_freq_interpolator, dim=-2)
        directivity_amplitude_response = torch.exp(directivity_response)


        # Combine frequency responses
        frequency_response = directivity_amplitude_response * reflection_frequency_response
        
        

        """
        Introducing delays ####################################################
        """

        #max_delay = max(loc.delays) + 100 ############### +100 to not lose information after toa perturb


        ################################################################################

        if self.toa_perturb:
            #noises = 7*torch.randn(n_paths, 1).to(self.device)################## 
            noises = 7*torch.randn(n_paths).to(self.device)################## 
        """
        ENTERING THE TIME DOMAIN
        """
        phases = hilbert_one_sided(safe_log(frequency_response), device=self.device)
        fx2 = frequency_response*torch.exp(1j*phases)
        sig = torch.fft.irfft(fx2)
        
        #new_freq_resp = torch.zeros(n_paths, int((max_delay + len(sig[0]) +1)/2), dtype=torch.complex64).to(self.device)
        new_freq_resp = torch.zeros(n_paths, int(self.RIR_length/2 + 1), dtype=torch.complex64).to(self.device)
        #new_freq_resp = torch.zeros(n_paths, int(max_delay + len(sig[0]))).to(self.device)#########################à
        
        new_window = torch.Tensor(
            scipy.fft.fftshift(scipy.signal.get_window("hamming", self.RIR_length, fftbins=False))).to(self.device) #needed to create a new window because the dimension is different
        
        '''
        for i in range(n_paths):
            if self.toa_perturb:
                delay = loc.delays[i] + torch.round(noises[i]).int()
            else:
                delay = loc.delays[i]
    
            del_pad_sig = torch.cat([torch.zeros(delay).to(self.device), sig[i], torch.zeros(self.RIR_length).to(self.device)])[:(self.RIR_length)]
            #del_pad_sig = torch.cat([torch.zeros(loc.delays[i]).to(self.device), sig, torch.zeros(self.RIR_length - loc.delays[i]).to(self.device)])


            factor = (2*self.nyq)/343
            del_pad_sig = del_pad_sig*(factor/(delay)) #attenuation proportional to path length
            
            del_pad_sig *= new_window

            """
            BACK TO THE FREQUENCY DOMAIN
            """
            
            new_freq_resp[i] = torch.fft.rfft(del_pad_sig)#############
            
            
            
            if not self.model_transmission:###why is this if inside the for cycle??
                new_freq_resp = new_freq_resp*paths_without_transmissions.reshape(-1,1).to(self.device)
        '''
            
        ###########################################
            # Calcolo dei ritardi
        if self.toa_perturb:
            delays = loc.delays.to(self.device) + torch.round(noises).int()
        else:
            delays = loc.delays.to(self.device)   # Shape: [n_paths]
        
        # Creazione del segnale con padding e ritardo
        del_pad_sig = torch.zeros((n_paths, self.RIR_length), device=self.device, dtype=torch.float64)

        # Assicurati che anche l'operazione con 'out' sia coerente
        sig = sig.to(torch.float64)  # Converti 'out' a float64
        
        # Poi esegui scatter_add_
        del_pad_sig.scatter_add_(
            1, 
            (delays[:, None] + torch.arange(sig.shape[-1], device=self.device)[None, :]).long(),
            (sig * (2 * self.nyq / 343) / delays[:, None]).to(torch.float64)
        )

        # Applicazione della finestra
        del_pad_sig *= new_window  # Assumendo che new_window abbia shape [RIR_length]

        """
        BACK TO THE FREQUENCY DOMAIN
        """
        new_freq_resp = torch.fft.rfft(del_pad_sig, dim=1)  # Calcolo simultaneo della rfft su tutti i percorsi

        # Gestione della trasmissione (evitare if nel loop)
        if not self.model_transmission:
            new_freq_resp *= paths_without_transmissions.reshape(-1, 1).to(self.device)
        ##########################################################      
            
        
        phases = new_freq_resp.angle()
        
        # Module downsample
        
        
        #might put this in the initial parameters
        pre_bp_freqs = torch.Tensor([32, 45, 63, 90, 125, 180, 250, 360, 500, 720, 1000, 1400, 2000, 2800, 4000, 5600, 8000, 12000, 16000, 20000])
        #pre_bp_freqs = torch.Tensor([32, 63, 125, 250,  500, 1000, 2000, 4000, 8000, 16000])
        
        pre_bp_freq_indices = torch.round(pre_bp_freqs*((int(self.RIR_length)/2+1)/self.nyq)).int() ##########################
        self.pre_bp_interpolator = get_interpolator(int(self.RIR_length/2 + 1), pre_bp_freq_indices).to(self.device)###########################
        
        downsample_indices = torch.round(pre_bp_freqs * ((len(new_freq_resp) - 1) / self.nyq)).int()

                
        downsampled_modules = new_freq_resp[:, downsample_indices].abs() 
        


        beampattern_abs_orientations = -loc.end_directions_normalized #outgoing directions needed to compute arctans
        

        #Make sure listener_forward and listener_left are orthogonal
        assert np.abs(np.dot(listener_forward, listener_left)) < 0.01

        listener_up = np.cross(listener_forward, listener_left)
        listener_basis = np.stack((listener_forward, listener_left, listener_up), axis=-1)

        #Compute Azimuths and Elevation
        listener_coordinates = beampattern_abs_orientations @ listener_basis
        paths_azimuths = -np.degrees(np.arctan2(listener_coordinates[:, 1], listener_coordinates[:, 0])) #sign - needed to consider clockwise azimuths
        paths_elevations = np.degrees(np.arctan(listener_coordinates[:, 2]/np.linalg.norm(listener_coordinates[:, 0:2],axis=-1)+1e-8))

        directional_freq_responses = initialize_directional_list(azimuths, elevations, self.RIR_length, self.device)
        n_orders = len (self.bp_ord_cut_freqs)

        cutoffs = self.bp_ord_cut_freqs.detach() #########detached for the moment (used to generate NaNs and are not so important now with module interpolation)


        freq_samples_contributions = initialize_directional_list(azimuths, elevations, [n_paths, len(pre_bp_freqs)], self.device) ####modules (10 samples)
       
        
        for j in range(len(pre_bp_freqs)):
                                
            bp_weights = calculate_weights_all_orders(pre_bp_freqs[j], paths_azimuths, paths_elevations, cutoffs, self.device) #
            pattern_max = beam_pattern(paths_azimuths, paths_elevations, bp_weights, n_orders, self.device)#normalization factor
                    
            for direction in freq_samples_contributions:
                direction['f_response'][:,j] = downsampled_modules[:,j] * beam_pattern(direction['angle'][0], direction['angle'][1], bp_weights, n_orders, self.device)/pattern_max ############is it ok to use complexes?
                
                

                    
        #ph = Func.interpolate(phases.unsqueeze(1), size = self.RIR_length, mode = 'linear').squeeze(1)########squeeze e unsqueeze needed beacause interpolate receives 3D
        
        
                
        for direction in directional_freq_responses:
            matching_direction = next((r for r in freq_samples_contributions if r['angle'] == direction['angle']), None)
            module = torch.sum(matching_direction['f_response'].unsqueeze(-1) * self.pre_bp_interpolator, dim=-2) #interpolation on same direction contributions    
                    
            
            signals = module*torch.exp(1j*phases) #add the corrispondent phase
            

            
            direction['f_response'] = torch.sum (signals, dim= 0)
            
            
            out_full = torch.fft.irfft(direction['f_response'])
            

            '''
            new_window = torch.Tensor(
            scipy.fft.fftshift(scipy.signal.get_window("hamming", len(out_full), fftbins=False))).to(self.device) #needed to create a new window because the dimension is different

            direction['t_response'] = out_full * new_window #########is the window needed??
            '''
            
            direction['t_response'] = out_full#####################
            
            
            direction['t_response']= F.fftconvolve(
                    self.source_response - torch.mean(self.source_response), direction['t_response'])[:self.RIR_length]
            direction['t_response'] = direction['t_response']*((self.sigmoid(self.decay)**self.times)) #[:len(r['t_response'])])################should I add the cut??


        return directional_freq_responses



    
    ##########################################################################################################################
        
    def render_late(self, loc):    
        """Renders the late-stage RIR. Future work may implement other ways of modeling the late-stage."""    
        if self.late_stage_model == "UniformResidual":
            late = self.RIR_residual
        else:
            raise ValueError("Invalid Residual Mode")
        return late

    def render_RIR(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        """Renders the RIR."""
        early = self.render_early(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        while torch.sum(torch.isnan(early)) > 0: # Check for numerical issues
            print("nan found - trying again")
            early = self.render_early(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        late = self.render_late(loc=loc)

        # Blend early and late stage together using spline
        self.spline = torch.sum(self.sigmoid(self.spline_values).view(self.n_spline,1)*self.IK, dim=0)
        RIR = late*self.spline + early*(1-self.spline)
        return RIR
    
    ###############################################################################
    
    def render_RIR_music_instrument(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        """Renders the RIR without the source impulse response (which is supposed to be included in the instrument dry sound recording)."""
        early = self.render_early_music_instrument(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        while torch.sum(torch.isnan(early)) > 0: # Check for numerical issues
            print("nan found - trying again")
            early = self.render_early_music_instrument(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        late = self.render_late(loc=loc)

        # Blend early and late stage together using spline
        self.spline = torch.sum(self.sigmoid(self.spline_values).view(self.n_spline,1)*self.IK, dim=0)
        RIR = late*self.spline + early*(1-self.spline)
        return RIR
    
    ###############################################################################################
    
    def render_RIR_omni(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        """Renders the RIR."""
        early = self.render_early_microphone_response(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        while torch.sum(torch.isnan(early)) > 0: # Check for numerical issues
            print("nan found - trying again")
            early = self.render_early_microphone_response(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        late = self.render_late(loc=loc)

        # Blend early and late stage together using spline
        self.spline = torch.sum(self.sigmoid(self.spline_values).view(self.n_spline,1)*self.IK, dim=0)
        RIR = late*self.spline + early*(1-self.spline)
        return RIR
    ###############################################################################
    
    def render_RIR_cardioid(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        """Renders the RIR."""
        early = self.render_early_cardioid(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        while torch.sum(torch.isnan(early)) > 0: # Check for numerical issues
            print("nan found - trying again")
            early = self.render_early(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        late = self.render_late(loc=loc) * 0.5 #part of the late response is attenuated due to  the cardioid polar characteristic (approximation based on Neumann KM184 characteristics)

        # Blend early and late stage together using spline
        self.spline = torch.sum(self.sigmoid(self.spline_values).view(self.n_spline,1)*self.IK, dim=0)
        RIR = late*self.spline + early*(1-self.spline)
        return RIR
    
    ####################################################################################

    def render_RIR_directional(self, loc, azimuths, elevations, source_axis_1=None, source_axis_2=None,listener_forward = np.array([0,1,0]), listener_left = np.array([-1,0,0])):
        """Renders the RIR."""
        directional_responses = self.render_early_directional(loc=loc, source_axis_1=source_axis_1, source_axis_2=source_axis_2, azimuths=azimuths, elevations=elevations, listener_forward=listener_forward, listener_left=listener_left)

        for r in directional_responses:
            while torch.sum(torch.isnan(r['t_response'])) > 0: # Check for numerical issues
                print("nan found - trying again")
                directional_responses = self.render_early_directional(loc=loc, source_axis_1=source_axis_1, source_axis_2=source_axis_2, azimuths=azimuths, elevations=elevations, listener_forward=listener_forward, listener_left=listener_left)

        late = self.render_late(loc=loc)
        self.spline = torch.sum(self.sigmoid(self.spline_values).view(self.n_spline,1)*self.IK, dim=0)

        signal_to_add = late * 0.33 #attenuation approximated based on a 3rd order hypercardioid beampattern
    
        for r in directional_responses:    
            
            r['t_response'] = signal_to_add*self.spline + r['t_response']*(1-self.spline)


        return directional_responses
    ###################################################################################

    

class ListenerLocation():
    """
    Class for a Listener Locations renderer.

    Constructor Parameters
    ----------
    source_xyz: np.array (3,)
        xyz location of the sound source in meters.
    listener_xyz: np.array (3,)
        xyz location of the listener location in meters.
    n_surfaces: number of surfaces
    reflections: list of list of int. 
        Indices of surfaces that each path reflects on.
    transmission: list of list of int. 
        Indices of surfaces that each path transmits through.
    delays: np.array (n_paths,)
        time delays in samples for each path.
    start_directions: np.array(n_paths, 3)
        vectors in the start directions of each path
    end_directions: np.array(n_paths, 3)
        vectors indicating the direction at which each path enters the listener.
    """
    def __init__(self,
                 source_xyz,
                 listener_xyz,
                 n_surfaces,
                 reflections,
                 transmissions,
                 delays,
                 start_directions,
                 end_directions = None,
                 #####################################################
                 rendering_method = None, mic_orientation = None, mic_0_gains = None, mic_180_loss = None, cardioid_exponents = None
                 ):

        self.source_xyz = source_xyz
        self.listener_xyz = listener_xyz
        self.reflection_mask = gen_counts(reflections,n_surfaces)
        self.transmission_mask = gen_counts(transmissions,n_surfaces)
        self.delays = torch.tensor(delays)
        self.start_directions_normalized = torch.tensor(
            start_directions/np.linalg.norm(start_directions, axis=-1).reshape(-1, 1))

        if end_directions is not None:
            self.end_directions_normalized = torch.tensor(
                end_directions/np.linalg.norm(end_directions, axis=-1).reshape(-1, 1))
        else:
            self.end_directions_normalized = None
            
        ######################################################################
        self.rendering_method = rendering_method
        self.mic_orientation = mic_orientation
        self.mic_0_gains = mic_0_gains 
        self.mic_180_loss = mic_180_loss
        self.cardioid_exponents = cardioid_exponents
        
        #########################################################################

def get_listener(source_xyz, listener_xyz, surfaces, load_dir=None, load_num=None,
                 speed_of_sound=343, max_order=5,  parallel_surface_pairs=None, max_axial_order=50,
                 ####################################################################################
                 rendering_method = None, mic_orientation = None, mic_0_gains = None, mic_180_loss = None, cardioid_exponents = None):
    """
    Function to get a ListenerLocation. If load_dir is provided, loads precomputed paths

    Parameters
    ----------
    source_xyz: (3,) array indicating the source's location
    listener_xyz: (3,) array indicating the listener's location
    surface: list of Surface, surfaces comprising room geometry
    load_dir: directory of precomputed paths, if None, traces from scratch.
    speed_of_sounds: in m/s
    max_order: maximum reflection order to trace to, if tracing from scratch.
    load_num: within the directory of precomputed paths, the index to load from.
    parallel_surface_pairs: list of list of int, surface pairs to do axial boosting, provided as indices in the 'surfaces' argument.
    max_axial_order: max reflection order for parallel surfaces    

    Returns
    -------
    ListenerLocation, characterizing the listener location in the room.
    """
    if load_dir is None: 
        # Tracing from Scratch
        reflections, transmissions, delays, start_directions, end_directions = (
            trace1.get_reflections_transmissions_and_delays(
            source=source_xyz, dest=listener_xyz, surfaces=surfaces, speed_of_sound=speed_of_sound,
            max_order=max_order,parallel_surface_pairs=parallel_surface_pairs, max_axial_order=max_axial_order)
        )

    else:
        # Loading precomputed paths
        print("Listener Loading From: " + load_dir)
        reflections = np.load(os.path.join(load_dir,"reflections/"+str(load_num)+".npy"), allow_pickle=True)
        transmissions = np.load(os.path.join(load_dir, "transmissions/"+str(load_num)+".npy"), allow_pickle=True)
        delays = np.load(os.path.join(load_dir, "delays/"+str(load_num)+".npy"))
        start_directions = np.load(os.path.join(load_dir, "starts/"+str(load_num)+".npy"))
        end_directions = np.load(os.path.join(load_dir, "ends/"+str(load_num)+".npy"))

    L = ListenerLocation(
        source_xyz=source_xyz,
        listener_xyz=listener_xyz,
        n_surfaces=len(surfaces),
        reflections=reflections,
        transmissions=transmissions,
        delays=delays,
        start_directions = start_directions,
        end_directions = end_directions,
        ############################################
        rendering_method = rendering_method, mic_orientation = mic_orientation, mic_0_gains = mic_0_gains, mic_180_loss = mic_180_loss, cardioid_exponents = cardioid_exponents)

    return L

def get_interpolator(n_freq_target, freq_indices):
    """Function to return a tensor that helps with efficient linear frequency interpolation"""

    device = freq_indices.device
    result = torch.zeros(len(freq_indices),n_freq_target)
    diffs = torch.diff(freq_indices).to(device)

    for i,index in enumerate(freq_indices):  
        if i==0:
            linterp = torch.cat((torch.ones(freq_indices[0]).to(device), (1-torch.arange(diffs[0]).to(device)/diffs[0]).to(device)))
            result[i,0:freq_indices[1]] = linterp
        elif i==len(freq_indices)-1:
            linterp = torch.cat(((torch.arange(diffs[i-1]).to(device)/diffs[i-1]).to(device), torch.ones(n_freq_target-freq_indices[i]).to(device)))
            result[i,freq_indices[i-1]:] = linterp
        else:
            linterp = torch.cat(((torch.arange(diffs[i-1]).to(device)/diffs[i-1]).to(device), (1-torch.arange(diffs[i]).to(device)/diffs[i]).to(device)))
            result[i,freq_indices[i-1]:freq_indices[i+1]] = linterp

    return result

def gen_counts(surface_indices, n_surfaces):
    """Generates a (n_paths, n_surfaces) 0-1 mask indicating reflections"""
    n_reflections = len(surface_indices)
    result = torch.zeros(n_reflections, n_surfaces)
    for i in range(n_reflections):
        for j in surface_indices[i]:
            result[i,j] += 1
    return result

def get_time_interpolator(n_target, indices):
    """Function to return a tensor that helps with efficient linear interpolation"""
    result = torch.zeros(len(indices),n_target)
    diffs = torch.diff(indices)

    for i,index in enumerate(indices):  
        if i == 0:
            linterp = torch.cat((torch.arange(indices[0])/indices[0], 1-torch.arange(diffs[0])/diffs[0]))
            result[i,0:indices[1]] = linterp
        elif i == len(indices)-1:
            linterp = torch.cat((torch.arange(diffs[i-1])/diffs[i-1], torch.ones(n_target-indices[i])))
            result[i,indices[i-1]:] = linterp
        else:
            linterp = torch.cat((torch.arange(diffs[i-1])/diffs[i-1], 1-torch.arange(diffs[i])/diffs[i]))
            result[i,indices[i-1]:indices[i+1]] = linterp

    return result

def hilbert_one_sided(x, device):
    """
    Returns minimum phases for a given log-frequency response x.
    Assume x.shape[-1] is ODD
    """
    N = 2*x.shape[-1] - 1
    Xf = torch.fft.irfft(x, n=N)
    h = torch.zeros(N).to(device)
    h[0] = 1
    h[1:(N + 1) // 2] = 2
    x = torch.fft.rfft(Xf * h)
    return torch.imag(x)




def safe_log(x, eps=1e-9):
    """Prevents Taking the log of a non-positive number"""
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log(safe_x)

###########################################################################

def initialize_directional_list_oldVersion(angular_sensitivities, signal_length, device):
    frequency_range_list = []
    for characteristic in angular_sensitivities:
        frequency_dict = dict()
        frequency_dict['frequency_range'] = characteristic['frequency_range']
        directional_responses = []
        used_angle = 180/(int(180/characteristic['angle']))
        azimuths = np.arange(0, 360, used_angle)
        elevations = np.arange(-90, 90+ used_angle, used_angle)
        for elevation in elevations:
            if (elevation == -90 or elevation == 90):
                direction_dict = dict()
                direction_dict['direction'] = [0, elevation]#-90 and +90 elevations are considered having azimuth=0
                direction_dict['response'] = torch.zeros(signal_length).to(device)
                directional_responses.append(direction_dict)
            else:   
                for azimuth in azimuths:
                    direction_dict = dict()
                    direction_dict['direction'] = [azimuth, elevation]
                    direction_dict['response'] = torch.zeros(signal_length).to(device)
                    directional_responses.append(direction_dict)

        frequency_dict['responses'] = directional_responses
        frequency_range_list.append(frequency_dict)

    return frequency_range_list

#############################################################################

######################################################################################################################à

def initialize_directional_list_for_beampattern_oldVersion(angular_sensitivity, n_freq_samples, device):

    frequency_responses_list = []
    used_angle = 180/(int(180/angular_sensitivity))
    azimuths = np.arange(0, 360, used_angle)
    elevations = np.arange(-90, 90+ used_angle, used_angle)
    for elevation in elevations:
        if (elevation == -90 or elevation == 90):
            direction_dict = dict()
            direction_dict['angle'] = [0, elevation]#-90 and +90 elevations are considered having azimuth=0
            direction_dict['f_response'] = torch.zeros(n_freq_samples, dtype=torch.complex64).to(device)#######is complex64 ok?
            frequency_responses_list.append(direction_dict)
        else:   
            for azimuth in azimuths:
                direction_dict = dict()
                direction_dict['angle'] = [azimuth, elevation]
                direction_dict['f_response'] = torch.zeros(n_freq_samples, dtype=torch.complex64).to(device)
                frequency_responses_list.append(direction_dict)

    return frequency_responses_list

########################################################################################################################

####################################################################################################
def initialize_directional_list(azimuths, elevations, n_freq_samples, device):#################method to use specified points (for example obtained with fibonacci distribution)

    frequency_responses_list = []
    for i in range(len(azimuths)):
        direction_dict = dict()
        direction_dict['angle'] = [azimuths[i], elevations[i]]
        direction_dict['f_response'] = torch.zeros(n_freq_samples, dtype=torch.complex64).to(device)############is complex64 ok?
        frequency_responses_list.append(direction_dict)

    return frequency_responses_list

#################################################################################################


################################################################

def sinc_filter(cutoff, fs=48000, num_taps=513, device = 'cpu'):
    """Create a sinc filter kernel."""
    t = torch.arange(-(num_taps - 1) / 2, (num_taps - 1) / 2 + 1,  device=device)
    t = t / fs
    return torch.where(t == 0, 2 * cutoff, torch.sin(2 * np.pi * cutoff * t) / (np.pi * t))

#################################################################

def bandpass_filter(low_cutoff, high_cutoff, fs= 48000, num_taps=513, device = 'cpu'):
    """Create a bandpass filter using sinc function."""
    # Low-pass filter with cutoff = high_cutoff
    hlp = sinc_filter(high_cutoff, fs, num_taps,  device=device)
    # High-pass filter with cutoff = low_cutoff
    hhp = sinc_filter(low_cutoff, fs, num_taps,  device=device)
    # Bandpass is the difference between the two
    hbp = hlp - hhp
    # Apply a window function, e.g., Hamming window, to improve performance
    window = torch.hamming_window(num_taps, periodic=False,  device=device)
    hbp *= window
    return hbp
########################################################################

def apply_bandpass_filter(signal, low_cutoff, high_cutoff, fs = 48000, num_taps=513):
    """Creates a bandpass filter and applies it to a signal using convolution."""
    device = signal.device
    filter_kernel = bandpass_filter(low_cutoff, high_cutoff, fs, num_taps, device).to(device)##################.to(device) may be
    # Add a batch dimension and channel dimension if necessary
    if signal.dim() == 1:
        signal = signal.unsqueeze(0).unsqueeze(0)
    elif signal.dim() == 2:
        signal = signal.unsqueeze(1)

    # Ensure the filter kernel has the right dimensions
    filter_kernel = filter_kernel.unsqueeze(0).unsqueeze(0)
    padding = (num_taps - 1) // 2
    # Convolve the signal with the filter kernel
    filtered_signal = nn.functional.conv1d(signal, filter_kernel, padding=padding)
    return filtered_signal.squeeze()
################################################################################

#trying to use beam pattern to compute the contributions of every path in every direction

###########################################################
#old method (if this is used, rendering is not differentiable anymore)
def calculate_weights(azimuth_incoming, elevation_incoming, l_max):
    """
    Compute the weights W_{lm} for the direction of arrival of the signal.
    
    :param azimuth_incoming: Azimuth angle of the direction of arrival (in degrees).
    :param elevation_incoming: Elevation angle of the direction of arrival (in degrees).
    :param l_max: Maximum order to consider for the spheric harmonics.
    :return: Dictionary of weights W_{lm}.
    """
    azimuth_incoming = - azimuth_incoming
    phi_0 = np.deg2rad(azimuth_incoming)
    theta_0 = np.deg2rad(90 - elevation_incoming)

    bp_weights = {}
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi_0, theta_0)
            bp_weights[(l, m)] = Y_lm
    
    return bp_weights

##########################################################

###########################################################
def calculate_weights_all_orders(frequency, azimuth_incoming, elevation_incoming, bp_orders_cutoffs, device):
    """
    Compute the weights W_{lm} for the direction of arrival of the signal.
    
    :param azimuth_incoming: Azimuth angle of the direction of arrival (in degrees).
    :param elevation_incoming: Elevation angle of the direction of arrival (in degrees).
    :param l_max: Maximum order to consider for the spheric harmonics.
    :return: Dictionary of weights W_{lm}.
    """
    azimuth_incoming = - azimuth_incoming
    phi_0 = np.deg2rad(azimuth_incoming)
    theta_0 = np.deg2rad(90 - elevation_incoming)

    l_max = len(bp_orders_cutoffs)

    bp_weights = {}
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi_0, theta_0).to(device)
            if l!=0:
                Y_lm *= sigmoid(frequency-bp_orders_cutoffs[l-1])
            bp_weights[(l, m)] = Y_lm
    
    return bp_weights

#################################################

####################################################################
def beam_pattern_old_version(azimuth, elevation, bp_weights, l_max):
    """
    Compute beam pattern in a specific direction.
    
    :param azimuth: Azimuth angle (in degrees).
    :param elevation: Elevation angle (in degrees).
    :param weights: Dictionary of weights W_{lm}.
    :param l_max: Maximum order of the considered spheric harmonics.
    :return: Amplitude of the beam pattern in the specified direction.
    """
    azimuth=azimuth
    phi = np.deg2rad(azimuth)
    theta = np.deg2rad(90 - elevation)

    pattern = 0.0
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)
            pattern += bp_weights[(l, m)] * Y_lm
    
    return torch.abs(pattern.cpu())

#################################################################

#################################################
def beam_pattern(azimuth, elevation, bp_weights, l_max, device):
    """
    Compute beam pattern in a specific direction.
    
    :param azimuth: Azimuth angle (in degrees).
    :param elevation: Elevation angle (in degrees).
    :param weights: Dictionary of weights W_{lm}.
    :param l_max: Maximum order of the considered spheric harmonics.
    :return: Amplitude of the beam pattern in the specified direction.
    """
    azimuth=azimuth
    phi = np.deg2rad(azimuth)
    theta = np.deg2rad(90 - elevation)

    pattern = 0.0
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)
            pattern += bp_weights[(l, m)].to('cpu') * Y_lm
    
    return torch.abs(pattern.cpu()).to(device)

#####################################################

def normalized_sph_harm(m, l, phi, theta): #wrong, scupy.special nis already normalizing spherical harmonics
    # Calcolare il fattore di normalizzazione
    normalization_factor = np.sqrt((2 * l + 1) / (4 * np.pi) * 
                                   np.math.factorial(l - abs(m)) / np.math.factorial(l + abs(m)))
    
    # Calcolare l'armonica sferica non normalizzata
    Y_lm = sph_harm(m, l, phi, theta)
    
    # Applicare la normalizzazione
    return Y_lm * normalization_factor

####################################################
def sigmoid(x, k = 0.01):
    return 1 / (1 + torch.exp(-x * k))
#######################################################

def fibonacci_sphere(n_samples):
    """Distributes n_samples on a unit fibonacci_sphere"""
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)

    for i in range(n_samples):
        y = 1 - (i / float(n_samples - 1)) * 2
        radius = math.sqrt(1 - y * y)

        theta = phi * i

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)

def calculate_weights_all_orders_batch(frequencies, azimuth_incoming, elevation_incoming, bp_orders_cutoffs, device):
    """
    Compute the weights W_{lm} for the direction of arrival of the signal in batch.
    
    :param frequencies: Tensor of frequencies (batch_size).
    :param azimuth_incoming: Tensor of azimuth angles (batch_size).
    :param elevation_incoming: Tensor of elevation angles (batch_size).
    :param bp_orders_cutoffs: List of cutoffs for each spherical harmonic order.
    :param device: PyTorch device to use.
    :return: Dictionary of weights W_{lm} for each direction.
    """
    # Convert azimuth and elevation to radians
    azimuth_incoming = -azimuth_incoming  # Ensure proper azimuth orientation
    phi_0 = torch.deg2rad(azimuth_incoming).to(device)
    theta_0 = torch.deg2rad(90 - elevation_incoming).to(device)

    l_max = len(bp_orders_cutoffs)

    # Initialize an empty dictionary to store weights
    bp_weights = {}

    # Compute spherical harmonics for all l and m in batch
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Calculate the spherical harmonics in batch
            Y_lm = sph_harm(m, l, phi_0.cpu().numpy(), theta_0.cpu().numpy())
            Y_lm = torch.tensor(Y_lm, device=device)

            # Apply the sigmoid modulation for non-zero l
            if l != 0:
                Y_lm *= torch.sigmoid(frequencies - bp_orders_cutoffs[l-1]).to(device)

            # Store the result in the dictionary
            bp_weights[(l, m)] = Y_lm

    return bp_weights



def beam_pattern_batch(azimuths, elevations, bp_weights, l_max, device):
    """
    Compute the beam pattern in a specific direction for multiple directions (batch).
    
    :param azimuths: Tensor of azimuth angles (batch_size).
    :param elevations: Tensor of elevation angles (batch_size).
    :param bp_weights: Dictionary of spherical harmonic weights.
    :param l_max: Maximum order of the considered spherical harmonics.
    :param device: PyTorch device to use.
    :return: Tensor containing the beam pattern amplitudes for each direction.
    """
    # Convert azimuth and elevation to radians
    phi = torch.deg2rad(azimuths).to(device)
    theta = torch.deg2rad(90 - elevations).to(device)

    # Initialize the beam pattern result
    patterns = torch.zeros(azimuths.size(0), device=device)

    # Compute spherical harmonics for all l and m in batch
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Get the corresponding Y_lm values from the precomputed weights
            Y_lm = sph_harm(m, l, phi.cpu().numpy(), theta.cpu().numpy())
            Y_lm = torch.tensor(Y_lm, device=device)

            # Update the beam pattern with the weight for this order (l, m)
            patterns += bp_weights[(l, m)] * Y_lm

    return torch.abs(patterns)