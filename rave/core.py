import json
import os
from pathlib import Path
from random import random
from typing import Callable, Optional, Sequence, Union

import GPUtil as gpu
import librosa as li
import lmdb
import numpy as np
import pytorch_lightning as pl
import torch
import torch.fft as fft
import torch.nn as nn
import torchaudio
from einops import rearrange
from scipy.signal import lfilter
import math
import torch.nn.functional as F

from torchfcpe import spawn_bundled_infer_model

pitch_model = spawn_bundled_infer_model(device="cuda:0")

def search_for_config(folder):
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    configs = list(map(str, Path(folder).rglob("config.gin")))
    if configs != []:
        return os.path.abspath(os.path.join(folder, "config.gin"))
    configs = list(map(str, Path(folder).rglob("../config.gin")))
    if configs != []:
        return os.path.abspath(os.path.join(folder, "../config.gin"))
    configs = list(map(str, Path(folder).rglob("../../config.gin")))
    if configs != []:
        return os.path.abspath(os.path.join(folder, "../../config.gin"))
    else:
        return None

def init_weights_to_one(m):
    if isinstance(m, (nn.Linear)):  
        nn.init.constant_(m.weight, 1)        
        if m.bias is not None:               
            nn.init.constant_(m.bias, 0)

def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7

def harmonic_synth(pitch, amplitudes, sampling_rate):
    pitch = torch.clamp(pitch, min=1e-3)  # Avoid zero or negative frequencies
    pitch = torch.nan_to_num(pitch, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaNs/Infs
    
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal

def random_angle(min_f=20, max_f=8000, sr=24000):
    min_f = np.log(min_f)
    max_f = np.log(max_f)
    rand = np.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * np.pi * rand / sr
    return rand
    
def get_augmented_latent_size(latent_size: int, noise_augmentation: int):
    return latent_size + noise_augmentation

def pole_to_z_filter(omega, amplitude=.9):
    z0 = amplitude * np.exp(1j * omega)
    a = [1, -2 * np.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * np.real(z0), 1]
    return b, a

def random_phase_mangle(x, min_f, max_f, amp, sr):
    angle = random_angle(min_f, max_f, sr)
    b, a = pole_to_z_filter(angle, amp)
    return lfilter(b, a, x)


    return output

def get_ckpts(folder, name=None):
    ckpts = map(str, Path(folder).rglob("*.ckpt"))
    if name: 
        ckpts = filter(lambda e: mode in os.path.basename(str(e)), ckpts)
    ckpts = sorted(ckpts, key=os.path.getmtime)
    return ckpts

def get_versions(folder):
    ckpts = map(str, Path(folder).rglob("version_*"))
    ckpts = filter(lambda x: os.path.isdir(x), ckpts)
    return sorted(Path(dirpath).iterdir(), key=os.path.getmtime)

def search_for_config(folder):
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    configs = list(map(str, Path(folder).rglob("config.gin")))
    if configs != []:
        return os.path.abspath(os.path.join(folder, "config.gin"))
    configs = list(map(str, Path(folder).rglob("../config.gin")))
    if configs != []:
        return os.path.abspath(os.path.join(folder, "../config.gin"))
    configs = list(map(str, Path(folder).rglob("../../config.gin")))
    if configs != []:
        return os.path.abspath(os.path.join(folder, "../../config.gin"))
    else:
        return None

def search_for_run(run_path, name=None):
    if run_path is None: return None
    if ".ckpt" in run_path: return run_path
    ckpts = get_ckpts(run_path)
    if len(ckpts) != 0:
        return ckpts[-1]
    else:
        print('No checkpoint found')
    return None

def setup_gpu():
    return gpu.getAvailable(maxMemory=.05)

def get_beta_kl(step, warmup, min_beta, max_beta):
    if step > warmup: return max_beta
    t = step / warmup
    min_beta_log = np.log(min_beta)
    max_beta_log = np.log(max_beta)
    beta_log = t * (max_beta_log - min_beta_log) + min_beta_log
    return np.exp(beta_log)

def get_beta_kl_cyclic(step, cycle_size, min_beta, max_beta):
    return get_beta_kl(step % cycle_size, cycle_size // 2, min_beta, max_beta)

def get_beta_kl_cyclic_annealed(step, cycle_size, warmup, min_beta, max_beta):
    min_beta = get_beta_kl(step, warmup, min_beta, max_beta)
    return get_beta_kl_cyclic(step, cycle_size, min_beta, max_beta)

def n_fft_to_num_bands(n_fft: int) -> int:
    return n_fft // 2 + 1

def hinge_gan(score_real, score_fake):
    loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
    loss_dis = loss_dis.mean()
    loss_gen = -score_fake.mean()
    return loss_dis, loss_gen

def ls_gan(score_real, score_fake):
    loss_dis = (score_real - 1).pow(2) + score_fake.pow(2)
    loss_dis = loss_dis.mean()
    loss_gen = (score_fake - 1).pow(2).mean()
    return loss_dis, loss_gen

def nonsaturating_gan(score_real, score_fake):
    score_real = torch.clamp(torch.sigmoid(score_real), 1e-7, 1 - 1e-7)
    score_fake = torch.clamp(torch.sigmoid(score_fake), 1e-7, 1 - 1e-7)
    loss_dis = -(torch.log(score_real) + torch.log(1 - score_fake)).mean()
    loss_gen = -torch.log(score_fake).mean()
    return loss_dis, loss_gen

def get_minimum_size(model):
    N = 2**15
    device = next(iter(model.parameters())).device
    x = torch.randn(1, model.n_channels, N, requires_grad=True, device=device)
    z = model.encode(x)
    return int(x.shape[-1] / z.shape[-1])


@torch.enable_grad()
def get_rave_receptive_field(model, n_channels=1):
    N = 2**15
    model.eval()
    device = next(iter(model.parameters())).device

    for module in model.modules():
        if hasattr(module, 'gru_state') or hasattr(module, 'temporal'):
            module.disable()

    while True:
        x = torch.randn(1, model.n_channels, N, requires_grad=True, device=device)

        z = model.encode(x)
        z = model.encoder.reparametrize(z)[0]

        emb = torch.zeros(z.shape[0], 128, 1).to(z)
        #z = torch.cat((z, emb), dim=1)

        pitch = torch.rand(z.shape[0], 1, z.shape[-1])
        #amp = torch.rand(z.shape[0], 1, 2)
        z = torch.rand(x.shape[0], 66, 64)
        params = torch.rand(x.shape[0], 64, 103)
        
        y = model.decode(z, pitch, params, upp_factor=2048)

        y[0, 0, N // 2].backward()
        assert x.grad is not None, "input has no grad"

        grad = x.grad.data.reshape(-1)
        left_grad, right_grad = grad.chunk(2, 0)
        large_enough = (left_grad[0] == 0) and right_grad[-1] == 0
        if large_enough:
            break
        else:
            N *= 2
    left_receptive_field = len(left_grad[left_grad != 0])
    right_receptive_field = len(right_grad[right_grad != 0])
    model.zero_grad()

    for module in model.modules():
        if hasattr(module, 'gru_state') or hasattr(module, 'temporal'):
            module.enable()
    ratio = x.shape[-1] // z.shape[-1]
    rate = model.sr / ratio
    print(f"Compression ratio: {ratio}x (~{rate:.1f}Hz @ {model.sr}Hz)")
    return left_receptive_field, right_receptive_field

def valid_signal_crop(x, left_rf, right_rf):
    dim = x.shape[1]
    x = x[..., left_rf.item() // dim:]
    if right_rf.item():
        x = x[..., :-right_rf.item() // dim]
    return x

def relative_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return norm(x - y) / norm(x)

def mean_difference(target: torch.Tensor,
                    value: torch.Tensor,
                    norm: str = 'L1',
                    relative: bool = False):
    diff = target - value
    if norm == 'L1':
        diff = diff.abs().mean()
        if relative:
            diff = diff / target.abs().mean()
        return diff
    elif norm == 'L2':
        diff = (diff * diff).mean()
        if relative:
            diff = diff / (target * target).mean()
        return diff
    else:
        raise Exception(f'Norm must be either L1 or L2, got {norm}')

class MelScale(nn.Module):

    def __init__(self, sample_rate: int, n_fft: int, n_mels: int) -> None:
        super().__init__()
        mel = li.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        mel = torch.from_numpy(mel).float()
        self.register_buffer('mel', mel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = self.mel.type_as(x)
        y = torch.einsum('bft,mf->bmt', x, mel)
        return y

class MultiScaleSTFT(nn.Module):

    def __init__(self,
                 scales: Sequence[int],
                 sample_rate: int,
                 magnitude: bool = True,
                 normalized: bool = False,
                 num_mels: Optional[int] = None) -> None:
        super().__init__()
        self.scales = scales
        self.magnitude = magnitude
        self.num_mels = num_mels

        self.stfts = []
        self.mel_scales = []
        for scale in scales:
            self.stfts.append(
                torchaudio.transforms.Spectrogram(
                    n_fft=scale,
                    win_length=scale,
                    hop_length=scale // 4,
                    normalized=normalized,
                    power=None,
                ))
            if num_mels is not None:
                self.mel_scales.append(
                    MelScale(
                        sample_rate=sample_rate,
                        n_fft=scale,
                        n_mels=num_mels,
                    ))
            else:
                self.mel_scales.append(None)

        self.stfts = nn.ModuleList(self.stfts)
        self.mel_scales = nn.ModuleList(self.mel_scales)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = rearrange(x, "b c t -> (b c) t")
        stfts = []
        for stft, mel in zip(self.stfts, self.mel_scales):
            y = stft(x)
            if mel is not None:
                y = mel(y)
            if self.magnitude:
                y = y.abs()
            else:
                y = torch.stack([y.real, y.imag], -1)
            stfts.append(y)

        return stfts

class AudioDistanceV1(nn.Module):

    def __init__(self, multiscale_stft: Callable[[], nn.Module],
                 log_epsilon: float) -> None:
        super().__init__()
        self.multiscale_stft = multiscale_stft()
        self.log_epsilon = log_epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        stfts_x = self.multiscale_stft(x)
        stfts_y = self.multiscale_stft(y)
        distance = 0.

        for x, y in zip(stfts_x, stfts_y):
            logx = torch.log(x + self.log_epsilon)
            logy = torch.log(y + self.log_epsilon)

            lin_distance = mean_difference(x, y, norm='L2', relative=True)
            log_distance = mean_difference(logx, logy, norm='L1')

            distance = distance + lin_distance + log_distance

        return {'spectral_distance': distance}

class WeightedInstantaneousSpectralDistance(nn.Module):

    def __init__(self,
                 multiscale_stft: Callable[[], MultiScaleSTFT],
                 weighted: bool = False) -> None:
        super().__init__()
        self.multiscale_stft = multiscale_stft()
        self.weighted = weighted

    def phase_to_instantaneous_frequency(self,
                                         x: torch.Tensor) -> torch.Tensor:
        x = self.unwrap(x)
        x = self.derivative(x)
        return x

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:] - x[..., :-1]

    def unwrap(self, x: torch.Tensor) -> torch.Tensor:
        x = self.derivative(x)
        x = (x + np.pi) % (2 * np.pi)
        return (x - np.pi).cumsum(-1)

    def forward(self, target: torch.Tensor, pred: torch.Tensor):
        stfts_x = self.multiscale_stft(target)
        stfts_y = self.multiscale_stft(pred)
        spectral_distance = 0.
        phase_distance = 0.

        for x, y in zip(stfts_x, stfts_y):
            assert x.shape[-1] == 2

            x = torch.view_as_complex(x)
            y = torch.view_as_complex(y)

            # AMPLITUDE DISTANCE
            x_abs = x.abs()
            y_abs = y.abs()

            logx = torch.log1p(x_abs)
            logy = torch.log1p(y_abs)

            lin_distance = mean_difference(x_abs,
                                           y_abs,
                                           norm='L2',
                                           relative=True)
            log_distance = mean_difference(logx, logy, norm='L1')

            spectral_distance = spectral_distance + lin_distance + log_distance

            # PHASE DISTANCE
            x_if = self.phase_to_instantaneous_frequency(x.angle())
            y_if = self.phase_to_instantaneous_frequency(y.angle())

            if self.weighted:
                mask = torch.clip(torch.log1p(x_abs[..., 2:]), 0, 1)
                x_if = x_if * mask
                y_if = y_if * mask

            phase_distance = phase_distance + mean_difference(
                x_if, y_if, norm='L2')

        return {
            'spectral_distance': spectral_distance,
            'phase_distance': phase_distance
        }

class EncodecAudioDistance(nn.Module):

    def __init__(self, scales: int,
                 spectral_distance: Callable[[int], nn.Module]) -> None:
        super().__init__()
        self.waveform_distance = WaveformDistance(norm='L1')
        self.spectral_distances = nn.ModuleList(
            [spectral_distance(scale) for scale in scales])

    def forward(self, x, y):
        waveform_distance = self.waveform_distance(x, y)
        spectral_distance = 0
        for dist in self.spectral_distances:
            spectral_distance = spectral_distance + dist(x, y)

        return {
            'waveform_distance': waveform_distance,
            'spectral_distance': spectral_distance
        }

class WaveformDistance(nn.Module):

    def __init__(self, norm: str) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x, y):
        return mean_difference(y, x, self.norm)

class SpectralDistance(nn.Module):

    def __init__(
        self,
        n_fft: int,
        sampling_rate: int,
        norm: Union[str, Sequence[str]],
        power: Union[int, None],
        normalized: bool,
        mel: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mel:
            self.spec = torchaudio.transforms.MelSpectrogram(
                sampling_rate,
                n_fft,
                hop_length=n_fft // 4,
                n_mels=mel,
                power=power,
                normalized=normalized,
                center=False,
                pad_mode=None,
            )
        else:
            self.spec = torchaudio.transforms.Spectrogram(
                n_fft,
                hop_length=n_fft // 4,
                power=power,
                normalized=normalized,
                center=False,
                pad_mode=None,
            )

        if isinstance(norm, str):
            norm = (norm, )
        self.norm = norm

    def forward(self, x, y):
        x = self.spec(x)
        y = self.spec(y)

        distance = 0
        for norm in self.norm:
            distance = distance + mean_difference(y, x, norm)
        return distance

class ProgressLogger(object):

    def __init__(self, name: str) -> None:
        self.env = lmdb.open("status")
        self.name = name

    def update(self, **new_state):
        current_state = self.__call__()
        with self.env.begin(write=True) as txn:
            current_state.update(new_state)
            current_state = json.dumps(current_state)
            txn.put(self.name.encode(), current_state.encode())

    def __call__(self):
        with self.env.begin(write=True) as txn:
            current_state = txn.get(self.name.encode())
        if current_state is not None:
            current_state = json.loads(current_state.decode())
        else:
            current_state = {}
        return current_state

class LoggerCallback(pl.Callback):

    def __init__(self, logger: ProgressLogger) -> None:
        super().__init__()
        self.state = {'step': 0, 'warmed': False}
        self.logger = logger

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx) -> None:
        self.state['step'] += 1
        self.state['warmed'] = pl_module.warmed_up

        if not self.state['step'] % 100:
            self.logger.update(**self.state)

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, step_period: int = None, **kwargs):
        super().__init__(**kwargs)
        self.step_period = step_period 
        self.__counter = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.__counter += 1
        if self.step_period:
            if self.__counter % self.step_period == 0:
                filename = os.path.join(self.dirpath, f"epoch_{self.__counter}{self.FILE_EXTENSION}")
                self._save_checkpoint(trainer, filename)

def get_valid_extensions():
    import torchaudio
    backend = torchaudio.get_audio_backend()
    if backend in ["sox_io", "sox"]:
        return ['.'+f for f in torchaudio.utils.sox_utils.list_read_formats()]
    elif backend == "ffmpeg":
        return ['.'+f for f in torchaudio.utils.ffmpeg_utils.get_audio_decoders()]
    elif backend == "soundfile":
        return ['.wav', '.flac', '.ogg', '.aiff', '.aif', '.aifc']

# ------------------------------------------------

def extract_pitch(y, sr: int, block_size: int):
    f0_target_length=(y.shape[-1] // block_size)
    f0 = pitch_model.infer(y.unsqueeze(-1),
                         sr=sr,
                         decoder_mode='local_argmax',
                         threshold=0.006,
                         interp_uv=False,
                         f0_min=50,  # Minimum pitch
                         f0_max=2000,  # Maximum pitch
                         output_interp_target_length=f0_target_length)
    return f0


def z_score_normalization(contour, mean, std):
    return ((contour - mean) / std)


def scale_function(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7


def amp_to_impulse_response(amp, target_size: int):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


@torch.no_grad()
def a_weighted_loudness(x, sr: int, n_fft: int, block_size: int, win_length: int=None, window: str='hann', center: bool =False, to_linear: bool=False):
    # Support (Batch, Time) or (Time,) inputs
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        x = x.squeeze(1)

    if win_length is None:
        win_length = n_fft
        
    win_length = n_fft if win_length is None else win_length
    
    x_stft = torch.stft(x,
                        n_fft=n_fft,
                        hop_length=block_size,
                        win_length=win_length,
                        center=center,
                        return_complex=True)
    
    mag_stft = torch.abs(x_stft)
    num_freqs = mag_stft.shape[-1]
    freq_axis = torch.linspace(0, sr/2, num_freqs).to(x.device)
    
    f_sq = freq_axis ** 2
    const = torch.tensor([12194.217, 20.598997, 107.65265, 737.86223]).to(x.device) ** 2.0
    
    weights = 2.0 + 20.0 * (
        torch.log10(const[0])
        + 2 * torch.log10(f_sq)
        - torch.log10(f_sq + const[0])
        - torch.log10(f_sq + const[1])
        - 0.5 * torch.log10(f_sq + const[2])
        - 0.5 * torch.log10(f_sq + const[3])
    )
    
    weights = weights.view(1, 1, -1)
    
    perceptual_stft = 10 * torch.log10(mag_stft**2 + 1e-10) + weights
    loudness = torch.log10(torch.mean(torch.pow(10, perceptual_stft/10), dim=1) + 1e-5)

    if to_linear:
        loudness_norm = 10 ** (loudness / 20)
        return loudness, loudness_norm
    else:
        return loudness


def warmup_lambda(step, warmup_start, warmup_length, final_weight=0.5):
    if step < warmup_start:
        return 0.0 
    elif step < warmup_start + warmup_length:
        return final_weight * (step - warmup_start) / warmup_length
    else:
        return final_weight 


def threshold(periodicity: torch.Tensor, value: float=0.065):
    return periodicity > value


def upsample(signal, factor: int):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes: torch.Tensor, pitch: torch.Tensor, sampling_rate: int):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa


def entropy(logits):
    distribution = torch.nn.functional.softmax(logits, dim=1)
    return (
        1 + 1 / math.log(1440) * \
        (distribution * torch.log(distribution + 1e-7)).sum(dim=1))


def cents_to_bins(cents, quantize_fn=torch.floor):
    bins = quantize_fn(cents / 5.0).long()
    bins[bins < 0] = 0
    bins[bins >= 1440] = 1440 - 1
    return bins


def frequency_to_cents(frequency):
    return 1200 * torch.log2(frequency / 31.0)


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def bins_to_cents(bins):
    return 5.0 * bins


def cents(a, b):
    return 1200 * torch.log2(a / b)


def cents_to_frequency(cents):
    return 31.0 * 2 ** (cents / 1200)


def bins_to_frequency(bins):
    return cents_to_frequency(bins_to_cents(bins))
        

class PitchLoss(torch.nn.Module):
    def __init__(self,
                 pitch_bins: int = 1440,
                 gaussian_blur: bool = True,
                 loss_fn: str = 'categorical_cross_entropy',
                 fmin: float = 31.,
                 octave: int = 1200,
                 cents_per_bin: float = 5.):

        super().__init__()
        
        self.pitch_bins = pitch_bins
        self.gaussian_blur = gaussian_blur
        self.loss_fn = loss_fn
        
        self.fmin = fmin
        self.octave = octave
        self.cents_per_bin = cents_per_bin

        if gaussian_blur:
            cents = self.bins_to_cents(torch.arange(pitch_bins))
            self.register_buffer('cents', cents[:, None])

    def bins_to_cents(self, bins):
        return self.cents_per_bin * bins
        
    def bins_to_frequency(self, bins):
        return self.cents_to_frequency(self.bins_to_cents(bins))
    
    def cents_to_bins(self, cents, quantize_fn=torch.floor):
        bins = quantize_fn(cents / self.cents_per_bin).long()
        bins[bins < 0] = 0
        bins[bins >= self.pitch_bins] = self.pitch_bins - 1
        return bins
    
    def cents_to_frequency(self, cents):
        return self.fmin * 2 ** (cents / self.octave )
    
    def frequency_to_bins(self, frequency, quantize_fn=torch.floor):
        return self.cents_to_bins(self.frequency_to_cents(frequency), quantize_fn)
    
    def frequency_to_cents(self, frequency):
        return self.octave * torch.log2(frequency / self.fmin)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        
        bins = self.frequency_to_bins(target)
        voiced = target > 0
        bins = torch.where(voiced.to(logits.device),
                           bins.to(logits.device),
                           torch.randint(0, self.pitch_bins, bins.shape, dtype=torch.long).to(logits.device))
        
        logits = logits.permute(0, 2, 1).reshape(-1, self.pitch_bins)
        bins = bins.flatten().to(logits.device)
    
        if self.gaussian_blur:
            self.cents = self.cents.to(bins.device)
            distributions = torch.distributions.Normal(self.bins_to_cents(bins), 25)
            bins = torch.exp(distributions.log_prob(self.cents)).permute(1, 0)
            bins = bins / (bins.max(dim=1, keepdims=True).values + 1e-8)
        else:
            bins = torch.nn.functional.one_hot(bins, self.pitch_bins).float()
    
        if self.loss_fn == 'binary_cross_entropy':
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                bins)
    
        elif self.loss_fn == 'categorical_cross_entropy':
            return torch.nn.functional.cross_entropy(logits, bins)
        else:
            raise ValueError(f'Loss {self.loss_fn} is not implemented')