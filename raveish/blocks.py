from functools import partial
from typing import Callable, Optional, Sequence, Union, Tuple

import cached_conv as cc
import gin
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram
import torch.nn.functional as F
import math

from .core import amp_to_impulse_response, fft_convolve, mod_sigmoid, scale_function
from .core import remove_above_nyquist, upsample, threshold

@gin.configurable
def normalization(module: nn.Module, mode: str = 'identity'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        return weight_norm(module)
    else:
        raise Exception(f'Normalization mode {mode} not supported')

def exponential_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.sigmoid(x) ** np.log(10) + 1e-7


class SampleNorm(nn.Module):

    def forward(self, x):
        return x / torch.norm(x, 2, 1, keepdim=True)


class Residual(nn.Module):

    def __init__(self, module, cumulative_delay=0):
        super().__init__()
        additional_delay = module.cumulative_delay
        self.aligned = cc.AlignBranches(
            module,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x):
        x_net, x_res = self.aligned(x)
        return x_net + x_res


class ResidualLayer(nn.Module):

    def __init__(
        self,
        dim,
        kernel_size,
        dilations,
        cumulative_delay=0,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()
        net = []
        cd = 0
        for d in dilations:
            net.append(activation(dim))
            net.append(
                normalization(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        dilation=d,
                        padding=cc.get_padding(kernel_size, dilation=d),
                        cumulative_delay=cd,
                    )))
            cd = net[-1].cumulative_delay
        self.net = Residual(
            cc.CachedSequential(*net),
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)


class DilatedUnit(nn.Module):

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dilation: int,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)
    ) -> None:
        super().__init__()
        net = [
            activation(dim),
            normalization(
                cc.Conv1d(dim,
                          dim,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation,
                          ))),
            activation(dim),
            normalization(cc.Conv1d(dim, dim, kernel_size=1)),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[1].cumulative_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        layers = []
        cd = 0

        for dilations in dilations_list:
            layers.append(
                ResidualLayer(
                    dim,
                    kernel_size,
                    dilations,
                    cumulative_delay=cd,
                ))
            cd = layers[-1].cumulative_delay

        self.net = cc.CachedSequential(
            *layers,
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)


def normalize_dilations(dilations: Union[Sequence[int],
                                         Sequence[Sequence[int]]],
                        ratios: Sequence[int]):
    if isinstance(dilations[0], int):
        dilations = [dilations for _ in ratios]
    return dilations


# ------------------------------------------------

class HarmonicExcitationGenerator(torch.nn.Module):
    def __init__(self, 
                 sampling_rate: int, 
                 global_amp: float = 0.35,
                 block_size: int = 2048,
                 streaming: bool = False,
                 cat_harms: bool = False):
        
        super().__init__()
        self.sampling_rate = sampling_rate
        self.global_amp = global_amp
        self.block_size = block_size
        
        self.streaming = streaming
        self.cat_harms = cat_harms
        
        self.register_buffer("prev_phase", torch.zeros(1))
        
    def forward(self,
                pitch: torch.Tensor,
                harm_amps: torch.Tensor,
                periodicity: torch.Tensor,
                loudness: torch.Tensor,
                loudness_norm: torch.Tensor): # inputs = [B, T, 1]

        if self.cat_harms:
            self.prev_phase = torch.zeros(pitch.shape[0], harm_amps.shape[-1], device=pitch.device)
        
        uv = threshold(periodicity).to(pitch.device)
        pitch = torch.clamp(pitch, min=1e-3)
        pitch = torch.nan_to_num(pitch, nan=0.0, posinf=0.0, neginf=0.0) * uv
        
        harm_amps = remove_above_nyquist(
            harm_amps,
            pitch,
            self.sampling_rate,
        )

        harm_amps /= harm_amps.sum(-1, keepdim=True)

        harm_amps = upsample(harm_amps, self.block_size)
        pitch = upsample(pitch, self.block_size)
        ap = upsample((1.0 - periodicity) * uv, self.block_size)
        
        loudness = upsample(loudness, self.block_size)
        loudness_norm = upsample(loudness_norm, self.block_size)

        n_harmonic = harm_amps.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)

        if self.streaming:
            omega = omega + self.prev_phase
            if self.cat_harms:
                self.prev_phase.copy_(omega[0, -1, :] % (2 * math.pi))
            else:
                self.prev_phase.copy_(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
        
        if self.cat_harms:
            signal = (torch.sin(omegas) * harm_amps)
        else:
            signal = (torch.sin(omegas) * harm_amps).sum(-1, keepdim=True)
      
        noise = (torch.rand(ap.shape).to(ap) * 2 - 1) * self.global_amp
        noise = (noise * 0.1) * ap

        if self.cat_harms:
            signal = torch.cat((signal, noise), dim=-1)
        else:
            signal = signal * self.global_amp
        
        signal = (signal + noise) * loudness
        
        return signal.transpose(2, 1), noise.transpose(2, 1), loudness_norm.transpose(2, 1)



class AddUpDownSampling(nn.Module):

    def __init__(self, harm_channels, channels, kernel_size, net_delay):
        super().__init__()
        
        self.ex_conv = cc.Conv1d(harm_channels,
                                 channels,
                                 kernel_size=kernel_size*2,
                                 stride=kernel_size,
                                 padding=cc.get_padding(kernel_size*2))

        sine_delay = self.ex_conv.cumulative_delay
        delays = [net_delay, sine_delay]

        max_delay = max(delays)

        self.paddings = nn.ModuleList([
            cc.CachedPadding1d(p, crop=True)
            for p in map(lambda f: max_delay - f, delays)
        ])

        self.cumulative_delay = max_delay

    def forward(self, x, ex):
        delayed_x = self.paddings[0](x)

        ex_down = self.ex_conv(ex)
        delayed_ex = self.paddings[1](ex_down)

        output = delayed_x + delayed_ex
        return output


class AmpBlock(nn.Module):

    def __init__(
        self,
        in_size,
        hidden_size,
        n_layers,
        output_size
    ):
        super().__init__()

        channels = [in_size] + (n_layers) * [hidden_size]

        net = []
        
        for i in range(n_layers):
            net.append(nn.Linear(channels[i], channels[i + 1]))
            net.append(nn.LayerNorm(channels[i + 1]))
            net.append(nn.LeakyReLU())

        net.append(nn.Linear(channels[i + 1], output_size))
        net.append(nn.Sigmoid())

        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        x = self.net(x)
        return x


class FiLM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FiLM, self).__init__()

        self.relu = torch.nn.LeakyReLU(0.2)

        self.scale_conv = normalization(cc.Conv1d(in_channels=in_channels,
                                                  out_channels=in_channels,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=cc.get_padding(3)))
        
        self.shift_conv = normalization(cc.Conv1d(in_channels=in_channels,
                                                  out_channels=in_channels,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=cc.get_padding(3)))

    def forward(self, x):
        x = self.relu(x)
        scale, shift = self.scale_conv(x), self.shift_conv(x)
        return scale, shift


class FeatureWiseAffine(nn.Module):
    def __init__(self):
        super(FeatureWiseAffine, self).__init__()

    def forward(self, x, scale, shift):
        outputs = scale * x + shift
        return outputs


class AlignAndFiLM(nn.Module):

    def __init__(self, channels, upsample_delay, downsample_delay):
        super().__init__()

        self.film1 = FiLM(channels, channels)
        self.film2 = FiLM(channels, channels)
        self.fw_affine = FeatureWiseAffine()

        delays = [upsample_delay, downsample_delay, downsample_delay] 
        max_delay = max(delays)

        self.paddings = nn.ModuleList([
            cc.CachedPadding1d(p, crop=True) 
            for p in (max_delay - f for f in delays)
        ])

        self.cumulative_delay = max_delay

    def forward(self, x, cond1, cond2):
        delayed_x = self.paddings[0](x)
        
        delayed_cond1 = self.paddings[1](cond1)
        delayed_cond2 = self.paddings[2](cond2)

        scale1, shift1 = self.film1(delayed_cond1)
        scale2, shift2 = self.film2(delayed_cond2)

        output = self.fw_affine(delayed_x, (scale1 + scale2), (shift1 + shift2))

        return output


class GeneratorV2SineV2(nn.Module):

    def __init__(
        self,
        data_size: int,
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        kernel_size: int,
        sampling_rate: int,
        n_coeffs: int,
        dilations: Sequence[int],
        harm_channels: int = 1,
        streaming: bool = False,
        keep_dim: bool = False,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        n_channels: int = 1,
        amplitude_modulation: bool = False,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        adain: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        dilations_list = normalize_dilations(dilations, ratios)[::-1]
        ratios = ratios[::-1]

        if keep_dim:
            num_channels = np.prod(ratios) * capacity
        else:
            num_channels = 2**len(ratios) * capacity

        if harm_channels > 1:
            self.ex_generator = HarmonicExcitationGenerator(sampling_rate=sampling_rate, streaming=streaming, cat_harms=True)
        else:
            self.ex_generator = HarmonicExcitationGenerator(sampling_rate=sampling_rate, streaming=streaming)

        self.sampling_rate = sampling_rate

        self.conditioning_stages = [2, 6, 11, 16]

        downsampling_channels = []

        net = []

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size))

        net.append(
            normalization(
                cc.Conv1d(
                    latent_size,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                )), )

        for r, dilations in zip(ratios, dilations_list):
            # ADD UPSAMPLING UNIT
            if keep_dim:
                out_channels = num_channels // r
            else:
                out_channels = num_channels // 2
            net.append(activation(num_channels))
            net.append(
                normalization(
                    cc.ConvTranspose1d(num_channels,
                                       out_channels,
                                       2 * r,
                                       stride=r,
                                       padding=r // 2)))
            
            downsampling_channels.append(out_channels)

            num_channels = out_channels

            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                if adain is not None:
                    net.append(adain(num_channels))
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )))

        net.append(activation(num_channels))

        waveform_module = normalization(
            cc.Conv1d(
                num_channels,
                data_size * 2 if amplitude_modulation else data_size,
                kernel_size=kernel_size * 2 + 1,
                padding=cc.get_padding(kernel_size * 2 + 1),
            ))

        net.append(waveform_module)

        self.net = cc.CachedSequential(*net)
        self.amplitude_modulation = amplitude_modulation

        # ----- EXCITATION & NOISE DOWNSAMPLER NETS ------
        
        self.align_net = nn.ModuleList()
        for channels, stage in zip(downsampling_channels, self.conditioning_stages):
            self.align_net.append(AlignAndFiLM(channels, self.net[stage].cumulative_delay, 0))

        downsampling_channels = downsampling_channels[::-1]
        down_ratios = [16, 4, 4, 4]

        excitation_net = []
        loudness_net = []

        in_channels = 1

        for out_channels, kernel_size in zip(downsampling_channels, down_ratios):
            excitation_net.append(normalization(
                cc.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * kernel_size,
                    stride=kernel_size,
                    padding=cc.get_padding(2 * kernel_size, kernel_size))))
            
            loudness_net.append(normalization(
                cc.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * kernel_size,
                    stride=kernel_size,
                    padding=cc.get_padding(2 * kernel_size, kernel_size))))
    
            in_channels = out_channels

        self.excitation_net = cc.CachedSequential(*excitation_net)
        self.loudness_net = cc.CachedSequential(*loudness_net)

    def forward(self,
                x: torch.Tensor,
                f0: torch.Tensor,
                amplitudes: torch.Tensor,
                loudness_norm: torch.Tensor,
                loudness_lin: torch.Tensor,
                periodicity: torch.Tensor,
                upp_factor: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        har_source, noise, loudness_norm = self.ex_generator(f0, amplitudes, periodicity, loudness_lin, loudness_norm) # inputs = [B, T, c]
        
        har_output = har_source
        
        harmonic_downsampled = []
        loudness_downsampled = []

        for i, (layer_ex, layer_loudness) in enumerate(zip(self.excitation_net, self.loudness_net)):
            har_source = layer_ex(har_source)
            harmonic_downsampled.append(har_source)
            loudness_norm = layer_loudness(loudness_norm)
            loudness_downsampled.append(loudness_norm)

        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 2:
                x = self.align_net[0](x, harmonic_downsampled[3], loudness_downsampled[3])
            elif i == 6:
                x = self.align_net[1](x, harmonic_downsampled[2], loudness_downsampled[2])
            elif i == 11:
                x = self.align_net[2](x, harmonic_downsampled[1], loudness_downsampled[1])
            elif i == 16:
                x = self.align_net[3](x, harmonic_downsampled[0], loudness_downsampled[0])

        if self.amplitude_modulation:
            x, amplitude = x.split(x.shape[1] // 2, 1)
            x = x * torch.sigmoid(amplitude)

        return torch.tanh(x), har_output, noise

    def set_warmed_up(self, state: bool):
        pass