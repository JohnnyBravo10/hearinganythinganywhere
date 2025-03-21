import numpy as np
import torch
import scipy.signal as signal

torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def safe_log(x, eps=1e-7):
    """
    Avoid taking the log of a non-positive number
    """
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log(safe_x)


def get_stft(x, n_fft, hop_length=None):
    """
    Returns the stft of x.
    """
    return torch.stft(x.to(device),  ########## original didn't have .to(device)
                      n_fft=n_fft,
                      hop_length = hop_length,
                      window=torch.hann_window(n_fft).to(device),
                      return_complex=False)




"""
Training Losses
"""
def L1_and_Log(x,y, n_fft=512, hop_length=None, eps=1e-6):
    """
    Computes spectral L1 plus log spectral L1 loss

    Parameters
    ----------
    x: first audio waveform(s), torch.tensor
    y: second audio waveform(s), torch.tensor
    n_fft: n_fft for stft
    hop_length: stft hop length
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss (float)
    """
    est_stft = get_stft(x, n_fft=n_fft,hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft,hop_length=hop_length)
    
    assert est_stft.shape == ref_stft.shape    
    est_amp = torch.sqrt(est_stft[..., 0]**2 + est_stft[..., 1]**2 + eps)
    ref_amp = torch.sqrt(ref_stft[..., 0]**2 + ref_stft[..., 1]**2 + eps)

    result = torch.mean(torch.abs(safe_log(est_amp)-safe_log(ref_amp))) + torch.mean(torch.abs(est_amp-ref_amp))
    return result

def training_loss(x,y,cutoff=9000, eps=1e-6):
    """
    Training Loss

    Computes spectral L1 and log spectral L1 loss

    Parameters
    ----------
    x: first audio waveform(s), torch.tensor
    y: second audio waveform(s), torch.tensor
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss: float tensor
    """
    loss1 = L1_and_Log(x,y, n_fft=512, eps=eps)
    loss2 = L1_and_Log(x,y, n_fft=1024, eps=eps)
    loss3 = L1_and_Log(x,y, n_fft=2048, eps=eps)
    loss4 = L1_and_Log(x,y, n_fft=4096, eps=eps)
    tiny_hop_loss = L1_and_Log(x[...,:cutoff], y[...,:cutoff], n_fft=256, eps=eps, hop_length=1)
    return loss1 + loss2 + loss3 + loss4 + tiny_hop_loss

###############################################################################################################
###old method
def simplified_loss(x,y,cutoff=9000, eps=1e-6):
    """
    Training Loss

    Computes spectral L1 and log spectral L1 loss only with two windows

    Parameters
    ----------
    x: first audio waveform(s), torch.tensor
    y: second audio waveform(s), torch.tensor
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss: float tensor
    """
    loss1 = L1_and_Log(x,y, n_fft=512, eps=eps)
    loss4 = L1_and_Log(x,y, n_fft=4096, eps=eps)
    return loss1 + loss4 
#####################################################################################################################

######################################################################################################################
def training_loss_directional_0_old_version(x,y, cutoff =9000, eps=1e-6):
    """
    Training Loss considering directionality

    Computes spectral L1 and log spectral L1 loss for each direction

    Parameters
    ----------
    x: list of dictionaries (one for each frequency_range), torch.tensor
    y: lòist of dictionaries (one for each frequency_range), torch.tensor
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss: float tensor

    """

    assert len(x) == len(y), "Frequncy ranges lists have different sizes"
    loss = 0
    for interval in x:
        matching_interval = next((i for i in y if i['frequency_range'] == interval['frequency_range']), None)

        assert matching_interval != None, "Different frequency ranges"
        assert len(interval['responses']) == len(matching_interval['responses']), "Different resolutions"
        
        for r in interval['responses']:
            matching_r = next(i for i in matching_interval['responses'] if (i['direction'][0] == r['direction'][0] and i['direction'][1] == r['direction'][1]))
            loss += training_loss(r['response'], matching_r['response'], cutoff=cutoff, eps=eps)


    return loss

#############################################################################################################################

#############################################################################################################################
def training_loss_directional(x,y, cutoff =9000, eps=1e-6):
    """
    Training Loss considering directionality

    Computes spectral L1 and log spectral L1 loss for each direction

    Parameters
    ----------
    x: list of dictionaries (one for each direction), torch.tensor
    y: lòist of dictionaries (one for each direction), torch.tensor
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss: float tensor

    """

    assert len(x) == len(y), "Different angular resolutions"
    loss = 0
    for r in x:
        matching_r = next((i for i in y if i['angle'] == r['angle']), None)
        loss += training_loss(r['t_response'], matching_r['t_response'], cutoff=cutoff, eps=eps)


    return loss

##############################################################################################################################

#############################################################################################################################
def training_loss_directional_rates(x,y, cutoff =9000, eps=1e-6): 
    """
    Training Loss considering directionality

    Computes L1 between up-down, dx-sx and forward-behind power rates (richiede che ci sia questo ordine in x e y)

    Parameters
    ----------
    x: list of dictionaries (one for each direction), torch.tensor
    y: list of dictionaries (one for each direction), torch.tensor
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss: float tensor

    """

    assert len(x) == len(y) == 6
    loss = 0

    up_x = torch.sqrt(torch.mean((x[0]['t_response'])**2))
    down_x = torch.sqrt(torch.mean((x[1]['t_response'])**2))
    dx_x = torch.sqrt(torch.mean((x[2]['t_response'])**2))
    sx_x = torch.sqrt(torch.mean((x[3]['t_response'])**2))
    forward_x = torch.sqrt(torch.mean((x[4]['t_response'])**2))
    behind_x = torch.sqrt(torch.mean((x[5]['t_response'])**2))

    up_y = torch.sqrt(torch.mean((y[0]['t_response'])**2))
    down_y = torch.sqrt(torch.mean((y[1]['t_response'])**2))
    dx_y = torch.sqrt(torch.mean((y[2]['t_response'])**2))
    sx_y = torch.sqrt(torch.mean((y[3]['t_response'])**2))
    forward_y = torch.sqrt(torch.mean((y[4]['t_response'])**2))
    behind_y = torch.sqrt(torch.mean((y[5]['t_response'])**2))

    loss += torch.abs(up_x/down_x - up_y/down_y) + torch.abs(dx_x/sx_x - dx_y/sx_y) + torch.abs(forward_x/behind_x - forward_y/behind_y)


    return loss

##############################################################################################################################

#############################################################################################################################
def training_loss_directional_with_decay(x,y, cutoff =9000, eps=1e-6, l = 10):
    """
    Training Loss considering directionality

    Computes spectral L1, log spectral L1 and log L1 of the decay curves for each direction

    Parameters
    ----------
    x: list of dictionaries (one for each direction), torch.tensor
    y: lòist of dictionaries (one for each direction), torch.tensor
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.
    l: weight of decay loss
    
    Returns
    -------
    loss: float tensor

    """

    assert len(x) == len(y), "Different angular resolutions"
    loss = 0
    for r in x:
        matching_r = next((i for i in y if i['angle'] == r['angle']), None)
        loss += training_loss(r['t_response'], matching_r['t_response'], cutoff=cutoff, eps=eps)
        
        loss += l * torch.mean(torch.abs(safe_log(decay_curve(r['t_response']))-safe_log(decay_curve(matching_r['t_response']))))


    return loss

##############################################################################################################################

##############################################################################################################################

def training_loss_with_decay(x,y,cutoff=9000, eps=1e-6, l = 10): #########
    """
    Training Loss

    Computes spectral L1, log spectral L1 and log L1 of the decay curves

    Parameters
    ----------
    x: first audio waveform(s), torch.tensor
    y: second audio waveform(s), torch.tensor
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.
    l: weight of decay loss 
    
    Returns
    -------
    loss: float tensor
    """
    loss1 = L1_and_Log(x,y, n_fft=512, eps=eps)
    loss2 = L1_and_Log(x,y, n_fft=1024, eps=eps)
    loss3 = L1_and_Log(x,y, n_fft=2048, eps=eps)
    loss4 = L1_and_Log(x,y, n_fft=4096, eps=eps)
    tiny_hop_loss = L1_and_Log(x[...,:cutoff], y[...,:cutoff], n_fft=256, eps=eps, hop_length=1)
    decay_loss = torch.mean(torch.abs(safe_log(decay_curve(x))-safe_log(decay_curve(y))))
    
    return loss1 + loss2 + loss3 + loss4 + tiny_hop_loss + l * decay_loss

##############################################################################################################################

##############################################################################################################################
def decay_curve(x, n_fft = 512, hop_length = None):


    # Compute the spectrogram with Hann window
    '''
    H = torch.abs(torch.stft(x.to(device),
                      n_fft=n_fft,
                      hop_length = hop_length,
                      window=torch.hann_window(n_fft).to(device),
                      return_complex=False))
    '''
    H = get_stft(x, n_fft, hop_length)
    E = torch.sum(H[..., 0]**2 + H[..., 1]**2, axis=0) 
    #print("H:", H)###############
    #print("H.shape", H.shape)
    # Compute energy for each time window
    #E = torch.sum(H**2, axis=0) 
    #print("E:", E)###############
    #print("E.shape", E.shape)
    
    # Compute the decay curve D(H)
    K = H.shape[1]
    D = torch.zeros(K-1) #last element is exluded to prevent NaN

    for k in range(K-1):
        D[k] = 1 + (E[k] / torch.sum(E[k+1:]))#.item()

    print("D:", D)
    return D
##############################################################################################################################

"""
Evaluation Metrics
"""

def log_L1_STFT(x,y, n_fft=512, eps=1e-6, hop_length=None):
    """
    Computes log spectral L1 loss

    Parameters
    ----------
    x: first audio waveform(s), torch.tensor
    y: second audio waveform(s), torch.tensor
    n_fft: n_fft for stft
    hop_length: stft hop length
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss, float tensor
    """
    est_stft = get_stft(x, n_fft=n_fft, hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft, hop_length=hop_length)
    
    assert est_stft.shape == ref_stft.shape 

    est_amp = torch.sqrt(est_stft[..., 0]**2 + est_stft[..., 1]**2 + eps)
    ref_amp = torch.sqrt(ref_stft[..., 0]**2 + ref_stft[..., 1]**2 + eps)
    result = torch.mean(torch.abs(safe_log(est_amp)-safe_log(ref_amp)))

    return result

def multiscale_log_l1(x,y, eps=1e-6):
    """Spectral Evaluation Metric"""
    loss = 0
    loss += log_L1_STFT(x,y, n_fft=64, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=128, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=256, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=512, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=1024, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=2048, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=4096, eps=eps)
    return loss

def env_loss(x, y, envelope_size=32, eps=1e-6):
    """Envelope Evaluation Metric. x,y are tensors representing waveforms."""
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    env1 = signal.convolve(x**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps
    env2 = signal.convolve(y**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps

    loss =  (np.mean(np.abs(np.log(env1) - np.log(env2))))
    
    return loss

def rt60_error(x, y, eps=1e-13):
    """Percentual error in RT60"""
    x = x.detach().cpu()
    y = y.detach().cpu()
    pred_rt60 = compute_rt60(x)
    gt_rt60 = compute_rt60(y)

    loss =  100 * torch.abs(pred_rt60 - gt_rt60)/(gt_rt60 + eps)
    
    return loss

def rt60_diff(x, y, eps=1e-6):
    """Difference in RT60"""
    x = x.detach().cpu()
    y = y.detach().cpu()
    pred_rt60 = compute_rt60(x)
    gt_rt60 = compute_rt60(y)

    loss =  torch.abs(pred_rt60 - gt_rt60)
    
    return loss

baseline_metrics = [multiscale_log_l1, env_loss, rt60_error, rt60_diff]

def LRE(x, y, n_fft = 1024, hop_length=None, eps=1e-6):
    """LRE - Binaural Evaluation."""
    est_stft = get_stft(x, n_fft=n_fft, hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft, hop_length=hop_length)

    assert est_stft.shape == ref_stft.shape    
    est_amp = torch.sqrt(est_stft[..., 0]**2 + est_stft[..., 1]**2 + eps)
    ref_amp = torch.sqrt(ref_stft[..., 0]**2 + ref_stft[..., 1]**2 + eps)
    dif = torch.sum(est_amp[1])/torch.sum(est_amp[0]) - torch.sum(ref_amp[1])/torch.sum(ref_amp[0])
    dif = dif ** 2

    return dif.item()


def compute_rt60(signal, sample_rate = 48000):
    """
    Calcola l'RT60 di un segnale dato usando PyTorch.
    
    Args:
        signal (torch.Tensor): Segnale audio 1D.
        sample_rate (int): Frequenza di campionamento del segnale.
        
    Returns:
        float: RT60 calcolato in secondi.
    """
    # Calcola l'energia del segnale
    energy = signal ** 2

    max_idx = torch.argmax(energy)

    # Taglia il segnale a partire dal massimo
    energy = energy[max_idx:]
    # Normalizza l'energia e converti in dB
    energy_db = 10 * torch.log10(energy / torch.max(energy))

    # Trova l'intervallo di decadimento (-5 dB a -65 dB)
    start_idx = (energy_db <= -5).nonzero(as_tuple=True)[0][0]
    end_idx = (energy_db <= -65).nonzero(as_tuple=True)[0][0]

    # Calcola il tempo corrispondente
    t_start = start_idx / sample_rate
    t_end = end_idx / sample_rate

    # RT60 è il tempo necessario per un decadimento di 60 dB
    rt60 = (t_end - t_start) * (60 / (65 - 5))

    return rt60