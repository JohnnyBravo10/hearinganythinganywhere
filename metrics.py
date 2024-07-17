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
    return torch.stft(x,
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

###########################################################
def training_loss_considering_directionality(x,y, cutoff =9000, eps=1e-6):
    """
    Training Loss considering directionality

    Computes spectral L1 and log spectral L1 loss for each direction

    Parameters
    ----------
    x: dictionary of (up to) 24 first audio waveform(s) (one for each direction of arrival), torch.tensor
    y: dictionary of (up to) 24 second audio waveform(s) (one for each direction of arrival), torch.tensor
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss: float tensor
    """

    loss = 0
    all_keys = set(x.keys()).union(y.keys())
    for key in all_keys:
        if key in x and key in y:
            loss += training_loss((x[key]), y[key], cutoff=cutoff, eps=eps)
        elif key not in x:
            loss += training_loss(torch.zeros_like(y[key]), y[key], cutoff=cutoff, eps=eps)
        elif key not in y:
            loss += training_loss(x[key], torch.zeros_like(x[key]), cutoff=cutoff, eps=eps)

    return loss

############################################################

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

baseline_metrics = [multiscale_log_l1, env_loss]

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