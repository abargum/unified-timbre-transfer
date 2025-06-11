import math, random
from typing import Dict

import torch
import julius
import torch.nn as nn
from torch_pitch_shift import pitch_shift, semitones_to_ratio, get_fast_shifts
import torchaudio.functional as F

def calculate_rms(samples):
    """
    Calculates the root mean square.

    Based on https://github.com/iver56/audiomentations/blob/master/audiomentations/core/utils.py
    """
    return torch.sqrt(torch.mean(torch.square(samples), dim=-1, keepdim=False))


def rms_normalize(samples):
    rms = samples.square().mean(dim=-1, keepdim=True).sqrt()
    return samples / (rms + 1e-8)

# Based on torch-audiomentations
def _gen_noise(num_samples, sample_rate, device, f_decay=None, rolloffs=None):
    """
    Generate colored noise with f_decay decay using torch.fft
    """
    noise = torch.normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
        (sample_rate,),
        device=device,
    )
    spec = torch.fft.rfft(noise)
    if f_decay is not None:
        mask = 1 / (
            torch.linspace(1, (sample_rate / 2) ** 0.5, spec.shape[0], device=device)
            ** f_decay
        )
    else:
        freqs = abs(
            torch.fft.fftfreq(n=noise.shape[-1], d=1 / sample_rate, device=device)[
                : noise.shape[-1] // 2 + 1
            ]
        )
        mask = torch.where(
            torch.logical_or(freqs < rolloffs[0], freqs > rolloffs[1]), 1.0, 0.0
        )
        mask *= torch.rand_like(mask)
    spec *= mask
    noise = rms_normalize(torch.fft.irfft(spec).unsqueeze(0)).squeeze()
    noise = torch.cat([noise] * int(math.ceil(num_samples / sample_rate)))
    return noise[:num_samples]

# Fast transforms for augmentation on the fly
class ComposeTransforms(nn.Module):
    def __init__(
        self, transforms: Dict[str, nn.Module], probs: Dict[str, float]
    ) -> None:
        super().__init__()
        self.transforms = nn.ModuleDict(transforms)
        self.probs = probs

    def forward(self, data: Dict[str, torch.Tensor]):
        for k in self.probs.keys():
            p = self.probs[k]
            t = self.transforms[k]
            if random.random() < p:
                data = t(data)
        return data


class SloppyPEQ(torch.nn.Module):
    def __init__(self,
                 sample_rate=44100,
                 num_filters=3,
                 freq_range=[100.0, 3000.0],
                 gain_range=[-24.0, 24.0]) -> None:
        super().__init__()
        
        self.sample_rate = sample_rate
        self.num_filters = num_filters
        self.freq_range = freq_range
        self.gain_range = gain_range

    def power_ratio(self, r: float, a: float, b: float):
        return a * math.pow((b / a), r) 

    def parametric_eq(self, audio: torch.Tensor):
        center_freqs = [self.power_ratio(float(z) / (self.num_filters), self.freq_range[0], self.freq_range[1])
                     for z in range(self.num_filters)]

        gains = [random.uniform(self.gain_range[0], self.gain_range[1]) for _ in range(self.num_filters)]
        gain_linear_values = [10 ** (gain_db / 20) for gain_db in gains]
        gain_linear_values.append(1.0)
        gain_linear_values = torch.Tensor(gain_linear_values).to(audio.device).view(-1, 1, 1)

        bands = julius.split_bands(audio, cutoffs=center_freqs, sample_rate=self.sample_rate)
        equalized_bands = bands * gain_linear_values
        equalized_signal = equalized_bands.sum(dim=0)

        return equalized_signal

    def forward(self, data: Dict[str, torch.Tensor]):
        transformed = data.copy()
        transformed["audio"] = self.parametric_eq(data["audio"])
        return transformed



class PEQAug(torch.nn.Module):
    def __init__(self,
                 sample_rate=44100,
                 num_filters=3,
                 freq_range=[100.0, 3000.0],
                 gain_range=[-24.0, 24.0],
                 q_range=[1.0, 5.0]) -> None:
        super().__init__()
        
        self.sample_rate = sample_rate
        self.num_filters = num_filters
        self.freq_range = freq_range
        self.gain_range = gain_range
        self.q_range = q_range

    def power_ratio(self, r: float, a: float, b: float):
        return a * math.pow((b / a), r) 

    def parametric_eq(self, audio: torch.Tensor):
        center_freqs = [self.power_ratio(float(z) / (self.num_filters), self.freq_range[0], self.freq_range[1])
                     for z in range(self.num_filters)]

        Qs = [self.power_ratio(random.uniform(0, 1), self.q_range[0], self.q_range[1])
              for _ in range(self.num_filters)]

        gains = [random.uniform(self.gain_range[0], self.gain_range[1]) for _ in range(self.num_filters)]
    
        for i in range(self.num_filters):
            audio = F.equalizer_biquad(audio,
                                       sample_rate=self.sample_rate,
                                       center_freq=center_freqs[i],
                                       gain=gains[i],
                                       Q=Qs[i])
        return audio

    def forward(self, data: Dict[str, torch.Tensor]):
        transformed = data.copy()
        transformed["audio"] = self.parametric_eq(data["audio"])
        return transformed
        

# Based on torch-audiomentations (MIT License)
class PitchAug(nn.Module):
    def __init__(self, sample_rate, shift_range=[-6, 6]) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self._fast_shifts = get_fast_shifts(
            sample_rate,
            lambda x: x >= semitones_to_ratio(shift_range[0])
            and x <= semitones_to_ratio(shift_range[1])
            and x != 1,
        )

    def forward(self, data: Dict[str, torch.Tensor]):
        # audio: (batch, n_samples), f0: (batch, n_frames, 1)
        x = data["audio"][:, None, :]
        # apply to whole batch
        shift = random.choice(self._fast_shifts)
        transformed = data.copy()
        transformed["audio"] = pitch_shift(x, shift, self.sample_rate).squeeze(1)
        # shifted f0
        if "f0" in data:
            transformed["f0"] = data["f0"] * float(shift)
        return transformed


class VolumeAug(nn.Module):
    def __init__(self, mult_range=[0.2, 5.0]) -> None:
        super().__init__()
        self.mult_range = mult_range

    def forward(self, data: Dict[str, torch.Tensor]):
        peak = abs(data["audio"]).max()
        # sample in logspace
        max_m = math.log(min(1 / (peak + 1e-5), self.mult_range[1]))
        min_m = math.log(self.mult_range[0])
        mult = math.exp(min_m + random.random() * (max_m - min_m))
        transformed = data.copy()
        transformed["audio"] = data["audio"] * mult
        if "volume" in data:
            transformed["volume"] = data["volume"] * mult
        return transformed

class AddNoise(nn.Module):
    def __init__(
        self,
        min_snr_in_db: float = 2.0,
        max_snr_in_db: float = 10.0,
        min_f_decay: float = -3.0,
        max_f_decay: float = 3.0,
        sample_rate: int = 48000,
        use_rolloff: bool = False,
    ) -> None:
        super().__init__()
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.min_f_decay = min_f_decay
        self.max_f_decay = max_f_decay
        self.sample_rate = sample_rate
        self.use_rolloff = use_rolloff

    def forward(self, data: Dict[str, torch.Tensor]):
        # TODO: This might be applied after the volume calculation, in which case the volume value is different
        samples = data["audio"]
        orig_ndim = samples.ndim
        is_torch = isinstance(samples, torch.Tensor)
        if not is_torch:
            samples = torch.from_numpy(samples)
        if orig_ndim == 2:
            samples = samples[None, :, :]
        batch_size, num_channels, num_samples = samples.shape

        snr_db = (self.max_snr_in_db - self.min_snr_in_db) * torch.rand(
            batch_size, device=samples.device
        ) + self.min_snr_in_db
        if self.use_rolloff:
            # prevent calculating rolloff on silence
            rolloffs = self.rolloffs(samples + torch.rand_like(samples) * 1e-5)
            self.rolloff_max = max(torch.max(rolloffs).item(), self.rolloff_max)
            self.rolloff_min = min(torch.min(rolloffs).item(), self.rolloff_min)
            # (batch_size, num_samples)
            noise = torch.stack(
                [
                    _gen_noise(
                        num_samples,
                        self.sample_rate,
                        samples.device,
                        rolloffs=[self.rolloff_min, self.rolloff_max],
                    )
                    for i in range(batch_size)
                ]
            )
        else:
            f_decay = (self.max_f_decay - self.min_f_decay) * torch.rand(
                batch_size, device=samples.device
            ) + self.min_f_decay
            # (batch_size, num_samples)
            noise = torch.stack(
                [
                    _gen_noise(
                        num_samples,
                        self.sample_rate,
                        samples.device,
                        f_decay=f_decay[i],
                    )
                    for i in range(batch_size)
                ]
            )

        # (batch_size, num_channels)
        noise_rms = calculate_rms(samples) / (10 ** (snr_db.unsqueeze(dim=-1) / 20))
        transformed = data.copy()
        transformed["audio"] = samples + noise_rms.unsqueeze(-1) * noise.view(
            batch_size, 1, num_samples
        ).expand(-1, num_channels, -1)
        if not is_torch:
            transformed["audio"] = transformed["audio"].numpy()
        if orig_ndim == 2:
            transformed["audio"] = transformed["audio"][0]
        return transformed


class LPF(nn.Module):
    def __init__(
        self, min_cutoff_freq=150.0, max_cutoff_freq=7500.0, sample_rate=48000
    ) -> None:
        super().__init__()
        self.min_cutoff_norm = min_cutoff_freq / sample_rate
        self.max_cutoff_norm = max_cutoff_freq / sample_rate
        self.sample_rate = sample_rate

    def forward(self, data: Dict[str, torch.Tensor]):
        import julius

        cutoff = (
            torch.rand(1).item() * (self.max_cutoff_norm - self.min_cutoff_norm)
            + self.min_cutoff_norm
        )
        transformed = data.copy()
        transformed["audio"] = julius.lowpass_filter(data["audio"], cutoff)
        return transformed