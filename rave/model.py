import math
from time import time
from typing import Callable, Optional, Iterable, Dict

import gin, pdb
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from sklearn.decomposition import PCA
from pytorch_lightning.trainer.states import RunningStage
import wandb
import os
import pickle
import random
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from rave.augmentations import ComposeTransforms, AddNoise, PitchAug, SloppyPEQ

import rave.core
from rave.core import extract_pitch, a_weighted_loudness, z_score_normalization, warmup_lambda
from rave.core import bins_to_frequency, entropy
from . import blocks
from .utils import perceptive 

mds_dims = 2
fullTimbreData = perceptive.get_perceptual_centroids(mds_dims, timbre_path='rave/utils/timbre.npy')
GLOB_LAB = list(fullTimbreData['instruments'])
GLOB_EMB = fullTimbreData['pos']

# Load data (deserialize)
with open('rave/utils/loudness_stats.pkl', 'rb') as file:
    loudness_dict = pickle.load(file)

_default_loss_weights = {
    'audio_distance': 1.,
    'multiband_audio_distance': 1.,
    'adversarial': 1.,
    'feature_matching': 20,
    'pitch_loss': 0.1
}

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


class WarmupCallback(pl.Callback):

    def __init__(self) -> None:
        super().__init__()
        self.state = {'training_steps': 0}

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:
        if self.state['training_steps'] >= pl_module.warmup:
            pl_module.warmed_up = True
        self.state['training_steps'] += 1

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)


class QuantizeCallback(WarmupCallback):

    def on_train_batch_(self, trainer, pl_module, batch,
                             batch_idx) -> None:

        if pl_module.warmup_quantize is None: return

        if self.state['training_steps'] >= pl_module.warmup_quantize:
            if isinstance(pl_module.encoder, blocks.DiscreteEncoder):
                pl_module.encoder.enabled = torch.tensor(1).type_as(
                    pl_module.encoder.enabled)
        self.state['training_steps'] += 1


@gin.configurable
class BetaWarmupCallback(pl.Callback):

    def __init__(self, initial_value: float = .2,
                       target_value: float = .2,
                       warmup_len: int = 1,
                       log: bool = True) -> None:
        super().__init__()
        self.state = {'training_steps': 0}
        self.warmup_len = warmup_len
        self.initial_value = initial_value
        self.target_value = target_value
        self.log_warmup = log

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:
        self.state['training_steps'] += 1
        if self.state["training_steps"] >= self.warmup_len:
            pl_module.beta_factor = self.target_value
            return

        warmup_ratio = self.state["training_steps"] / self.warmup_len

        if self.log_warmup: 
            beta = math.log(self.initial_value) * (1 - warmup_ratio) + math.log(
                self.target_value) * warmup_ratio
            pl_module.beta_factor = math.exp(beta)
        else:
            beta = warmup_ratio * (self.target_value - self.initial_value) + self.initial_value
            pl_module.beta_factor = min(beta, self.target_value)

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)


@torch.fx.wrap
def _pqmf_encode(pqmf, x: torch.Tensor):
    batch_size = x.shape[:-2]
    x_multiband = x.reshape(-1, 1, x.shape[-1])
    x_multiband = pqmf(x_multiband)
    x_multiband = x_multiband.reshape(*batch_size, -1, x_multiband.shape[-1])
    return x_multiband


@torch.fx.wrap
def _pqmf_decode(pqmf, x: torch.Tensor, batch_size: Iterable[int], n_channels: int):
    x = x.reshape(x.shape[0] * n_channels, -1, x.shape[-1])
    x = pqmf.inverse(x)
    x = x.reshape(*batch_size, n_channels, -1)
    return x


@gin.configurable
class RAVE(pl.LightningModule):

    def __init__(
        self,
        latent_size,
        sampling_rate,
        pitch_encoder,
        decoder,
        amp_block,
        discriminator,
        phase_1_duration,
        gan_loss,
        pitch_loss,
        valid_signal_crop,
        feature_matching_fun,
        num_skipped_features,
        audio_distance: Callable[[], nn.Module],
        multiband_audio_distance: Callable[[], nn.Module],
        n_bands: int = 16,
        balancer = None,
        weights: Optional[Dict[str, float]] = None,
        warmup_quantize: Optional[int] = None,
        pqmf: Optional[Callable[[], nn.Module]] = None,
        spectrogram: Optional[Callable] = None,
        update_discriminator_every: int = 2,
        n_channels: int = 1,
        input_mode: str = "pqmf",
        output_mode: str = "pqmf",
        audio_monitor_epochs: int = 1,
        # for retro-compatibility
        enable_pqmf_encode: Optional[bool] = None,
        enable_pqmf_decode: Optional[bool] = None,
        is_mel_input: Optional[bool] = None,
        loss_weights = None,
        with_pitch_loss = False,
        with_augmentation = False
    ):
        super().__init__()
        self.pqmf = pqmf(n_channels=n_channels)
        self.spectrogram = None
        if spectrogram is not None:
            self.spectrogram = spectrogram
        assert input_mode in ['pqmf', 'mel', 'raw']
        assert output_mode in ['raw', 'pqmf']
        self.input_mode = input_mode
        self.output_mode = output_mode
        # retro-compatibility
        if (enable_pqmf_encode is not None) or (enable_pqmf_decode is not None):
            self.input_mode = "pqmf" if enable_pqmf_encode else "raw"
            self.output_mode = "pqmf" if enable_pqmf_decode else "raw"
        if (is_mel_input) is not None:
            self.input_mode = "mel"
        if loss_weights is not None:
            weights = loss_weights
        assert weights is not None, "RAVE model requires either weights or loss_weights (depreciated) keyword"

        self.pitch_encoder = pitch_encoder()
        self.decoder = decoder(n_channels=n_channels)
        self.amp_block = amp_block()

        self.discriminator = discriminator(n_channels=n_channels)
    
        self.audio_distance = audio_distance()
        self.multiband_audio_distance = multiband_audio_distance()

        self.gan_loss = gan_loss
        self.pitch_loss = pitch_loss()

        self.register_buffer("latent_pca", torch.eye(latent_size))
        self.register_buffer("latent_mean", torch.zeros(latent_size))
        self.register_buffer("fidelity", torch.zeros(latent_size))

        self.latent_size = latent_size

        self.automatic_optimization = False

        # SCHEDULE
        self.warmup = phase_1_duration
        self.warmup_quantize = warmup_quantize
        self.weights = _default_loss_weights
        self.weights.update(weights)
        self.warmed_up = False

        # CONSTANTS
        self.sr = sampling_rate
        self.valid_signal_crop = valid_signal_crop
        self.n_channels = n_channels
        self.feature_matching_fun = feature_matching_fun
        self.num_skipped_features = num_skipped_features
        self.update_discriminator_every = update_discriminator_every
        
        self.with_pitch_loss = with_pitch_loss
        self.with_augmentation = with_augmentation

        self.eval_number = 0
        self.beta_factor = 0.2
        self.integrator = None

        self.register_buffer("receptive_field", torch.tensor([0, 0]).long())
        self.audio_monitor_epochs = audio_monitor_epochs

        self.step = 0
        self.block_size = 2048

        if not self.with_pitch_loss:
            self.pitch_encoder.load_state_dict(torch.load(f"rave/utils/noncaus2048_mb6.pth", weights_only=True))
            self.pitch_encoder.eval()
            
        if self.with_augmentation:
            add_noise = AddNoise(min_snr_in_db=5.0, max_snr_in_db=20.0, sample_rate=self.sr)
            parametric_eq = SloppyPEQ(sample_rate=self.sr, gain_range=[-15.0, 15.0])
            shift_pitch = PitchAug(sample_rate=self.sr)
    
            transforms = {"shift": shift_pitch, "peq": parametric_eq, "noise": add_noise}
            probabilities = {"shift": 0.35, "peq": 0.5, "noise": 0.5}
    
            self.transforms = ComposeTransforms(transforms=transforms, probs=probabilities)
        
    def configure_optimizers(self):
        gen_p = list(self.decoder.parameters())
        gen_p += list(self.amp_block.parameters())
        gen_p += list(self.pitch_encoder.parameters())

        dis_p = list(self.discriminator.parameters())

        gen_opt = torch.optim.Adam(gen_p, 1e-3, (.5, .9))
        dis_opt = torch.optim.Adam(dis_p, 1e-4, (.5, .9))

        return ({'optimizer': gen_opt,
                 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.LinearLR(gen_opt, start_factor=1.0, end_factor=0.1, total_iters=self.warmup)}},
                {'optimizer':dis_opt})

    def _mel_encode(self, x: torch.Tensor):
        batch_size = x.shape[:-2]
        x = self.spectrogram(x)[..., :-1]
        x = torch.log1p(x).reshape(*batch_size, -1, x.shape[-1])
        return x

    def decode(self, z, pitch, amp, loudness_norm, loudness_linear, periodicity):
        batch_size = z.shape[:-2]
        y, harm, noise = self.decoder(z, pitch, amp, loudness_norm, loudness_linear, periodicity, upp_factor=self.block_size)
        
        if self.output_mode == "pqmf":
            y = _pqmf_decode(self.pqmf, y, batch_size=batch_size, n_channels=self.n_channels)
        
        return y, harm, noise

    def forward(self, x, inst_id=None, p_mult=1.0):

        loudness_norm, loudness_linear = self.get_loudness(x, inst_id)

        x_multiband = _pqmf_encode(self.pqmf, x)
        pitch_logits = self.pitch_encoder(x_multiband[:, :6, :])
        pitch = torch.argmax(pitch_logits, dim=1)
        pitch = bins_to_frequency(pitch).unsqueeze(-1)
        periodicity = entropy(pitch_logits)

        indices = [GLOB_LAB.index(i) for i in inst_id]
        emb = torch.tensor(GLOB_EMB[indices], dtype=torch.float32).to(x.device)     

        emb = emb.unsqueeze(-1)
        amplitudes = self.amp_block(emb.transpose(2,1))
        emb = emb.repeat(1, 1, periodicity.shape[-1])

        pitch = pitch * p_mult
            
        y, harm, noise = self.decode(emb,
                                 pitch, 
                                 amplitudes,
                                 loudness_norm.transpose(2,1),
                                 loudness_linear.unsqueeze(-1),
                                 periodicity.unsqueeze(-1))

        return y, harm, amplitudes, noise

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self.lr_schedulers().step()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def split_features(self, features):
        feature_real = []
        feature_fake = []
        for scale in features:
            true, fake = zip(*map(
                lambda x: torch.split(x, x.shape[0] // 2, 0),
                scale,
            ))
            feature_real.append(true)
            feature_fake.append(fake)
        return feature_real, feature_fake

    def get_loudness(self, x, inst_id):
        loudness_mean = torch.tensor([loudness_dict[inst]['mean'] for inst in inst_id]).to(x.device) 
        loudness_std = torch.tensor([loudness_dict[inst]['std'] for inst in inst_id]).to(x.device) 
        loudness, loudness_linear = a_weighted_loudness(x, self.sr, n_fft=self.block_size, block_size=self.block_size, to_linear=True)
        loudness_norm = z_score_normalization(loudness, loudness_mean.unsqueeze(-1), loudness_std.unsqueeze(-1))
        return loudness_norm.unsqueeze(1), loudness_linear

    def training_step(self, batch, batch_idx):

        self.decoder.train()
        self.amp_block.train()
        
        p = Profiler()
        gen_opt, dis_opt = self.optimizers()

        # --- Process input ---
        x_raw = batch[0]
        inst_id = batch[1]
        
        x_raw.requires_grad = True

        batch_size = x_raw.shape[:-2]
        self.decoder.set_warmed_up(self.warmed_up)

        if self.with_augmentation:
            x_raw = (self.transforms({'audio': x_raw.squeeze(1)})['audio']).unsqueeze(1)

        # --- Extract Target Pitch and Embedding ---     
        pitch = extract_pitch(x_raw.squeeze(1), sr=self.sr, block_size=self.block_size).to(x_raw.device)
        
        indices = [GLOB_LAB.index(i) for i in inst_id]
        emb = torch.tensor(GLOB_EMB[indices], dtype=torch.float32).to(x_raw.device)    

        loudness_norm, loudness_linear = self.get_loudness(x_raw, inst_id)

        # --- Run encoders ---
        x_multiband = _pqmf_encode(self.pqmf, x_raw)

        if self.with_pitch_loss:
            pitch_logits = self.pitch_encoder(x_multiband[:, :6, :])
            pitch_loss = self.pitch_loss(pitch_logits, pitch.transpose(2,1))
            periodicity = entropy(pitch_logits)
            if self.warmed_up:
                pitch_logits = pitch_logits.detach()
        else:
            with torch.no_grad():
                pitch_logits = self.pitch_encoder(x_multiband[:, :6, :])
                periodicity = entropy(pitch_logits)
        
        # --- Run decoder ---
        emb = emb.unsqueeze(-1)
        amplitudes = self.amp_block(emb.transpose(2,1))
        emb = emb.repeat(1, 1, periodicity.shape[-1])
        
        y, harm, _ = self.decoder(emb,
                                  pitch, 
                                  amplitudes,
                                  loudness_norm.transpose(2,1),
                                  loudness_linear.unsqueeze(-1),
                                  periodicity.unsqueeze(-1),
                                  upp_factor=self.block_size)
        
        if self.output_mode == "pqmf":
            y_multiband = y
            y_raw = _pqmf_decode(self.pqmf, y, batch_size=batch_size, n_channels=self.n_channels)
        else:
            y_raw = y 
            y_multiband = _pqmf_encode(self.pqmf, y)

        # TODO this has been added for training with num_samples = 65536 samples, output padding seems to mess with output dimensions. 
        # this may probably conflict with cached_conv
        y_raw = y_raw[..., :x_raw.shape[-1]]
        y_multiband = y_multiband[..., :x_multiband.shape[-1]]

        p.tick('decode')

        if self.valid_signal_crop and self.receptive_field.sum():
            x_multiband = rave.core.valid_signal_crop(
                x_multiband,
                *self.receptive_field,
            )
            y_multiband = rave.core.valid_signal_crop(
                y_multiband,
                *self.receptive_field,
            )
        p.tick('crop')

        # DISTANCE BETWEEN INPUT AND OUTPUT
        distances = {}
        multiband_distance =  self.multiband_audio_distance(
            x_multiband, y_multiband)
        p.tick('mb distance')
        for k, v in multiband_distance.items():
            distances[f'multiband_{k}'] = self.weights['multiband_audio_distance'] * v

        fullband_distance = self.audio_distance(x_raw, y_raw)
        p.tick('fb distance')

        for k, v in fullband_distance.items():
            distances[f'fullband_{k}'] = self.weights['audio_distance'] *  v

        feature_matching_distance = 0.

        if self.warmed_up:  # DISCRIMINATION
            xy = torch.cat([x_raw, y_raw], 0)
            features = self.discriminator(xy)

            feature_real, feature_fake = self.split_features(features)

            loss_dis = 0
            loss_adv = 0

            pred_real = 0
            pred_fake = 0

            for scale_real, scale_fake in zip(feature_real, feature_fake):
                current_feature_distance = sum(
                    map(
                        self.feature_matching_fun,
                        scale_real[self.num_skipped_features:],
                        scale_fake[self.num_skipped_features:],
                    )) / len(scale_real[self.num_skipped_features:])

                feature_matching_distance = feature_matching_distance + current_feature_distance

                _dis, _adv = self.gan_loss(scale_real[-1], scale_fake[-1])

                pred_real = pred_real + scale_real[-1].mean()
                pred_fake = pred_fake + scale_fake[-1].mean()

                loss_dis = loss_dis + _dis
                loss_adv = loss_adv + _adv

            feature_matching_distance = feature_matching_distance / len(
                feature_real)

        else:
            pred_real = torch.tensor(0.).to(x_raw)
            pred_fake = torch.tensor(0.).to(x_raw)
            loss_dis = torch.tensor(0.).to(x_raw)
            loss_adv = torch.tensor(0.).to(x_raw)
        p.tick('discrimination')

        # COMPOSE GEN LOSS
        loss_gen = {}
        loss_gen.update(distances)
        p.tick('update loss gen dict')

        if self.with_pitch_loss:
            loss_gen['pitch_loss'] = pitch_loss

        if self.warmed_up:
            loss_gen['feature_matching'] = self.weights['feature_matching'] * feature_matching_distance
            loss_gen['adversarial'] = self.weights['adversarial'] * loss_adv

        # OPTIMIZATION
        if not (batch_idx %
                self.update_discriminator_every) and self.warmed_up:
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()
            p.tick('dis opt')
        else:
            gen_opt.zero_grad()
            loss_gen_value = 0.
            for k, v in loss_gen.items():
                loss_gen_value += v * self.weights.get(k, 1.)
            loss_gen_value.backward()
            gen_opt.step()

        # LOGGING
        self.log("beta_factor", self.beta_factor)

        if self.warmed_up:
            self.log("loss_dis", loss_dis)
            self.log("pred_real", pred_real.mean())
            self.log("pred_fake", pred_fake.mean())

        self.log_dict(loss_gen)
        p.tick('logging')

        wandb.log({
            "loss_dis": loss_dis,
            "loss_gen": loss_gen
        })

        self.step += 1

    def validation_step(self, batch, batch_idx):

        x = batch[0]
        inst_id = batch[1]

        pitch = extract_pitch(x.squeeze(1), sr=self.sr, block_size=self.block_size).to(x.device)
        indices = [GLOB_LAB.index(i) for i in inst_id]
        emb = torch.tensor(GLOB_EMB[indices], dtype=torch.float32).to(x.device)
        
        loudness_norm, loudness_linear = self.get_loudness(x, inst_id)
        
        x_multiband = _pqmf_encode(self.pqmf, x)

        pitch_logits = self.pitch_encoder(x_multiband[:, :6, :])
        periodicity = entropy(pitch_logits)
        
        emb = emb.unsqueeze(-1)
        amplitudes = self.amp_block(emb.transpose(2,1))
        emb = emb.repeat(1, 1, periodicity.shape[-1])
            
        y, harm, noise = self.decode(emb,
                                     pitch, 
                                     amplitudes,
                                     loudness_norm.transpose(2,1),
                                     loudness_linear.unsqueeze(-1),
                                     periodicity.unsqueeze(-1))

        if harm.shape[-1] > 1:
            harm = harm[:, 0, :].unsqueeze(1)

        distance = self.audio_distance(x, y)
        full_distance = sum(distance.values())

        if self.trainer is not None:
            self.log('validation', full_distance)
            wandb.log({
            "validation": full_distance})

        return torch.cat([x, y, harm, noise], -1)

    def validation_epoch_end(self, out):
    
        audio = list(map(lambda x: x.cpu(), out))

        y = torch.cat(audio, 0)[:8].reshape(-1).numpy()

        wandb.log({
                f"audio_val_{self.eval_number}":
                wandb.Audio(y,
                            caption="audio",
                            sample_rate=self.sr)
            })
        
        if self.integrator is not None:
            y = self.integrator(y)
        self.logger.experiment.add_audio("audio_val", y, self.eval_number,
                                        self.sr)
        self.eval_number += 1

    def on_fit_start(self):
        tb = self.logger.experiment

        config = gin.operative_config_str()
        config = config.split('\n')
        config = ['```'] + config + ['```']
        config = '\n'.join(config)
        tb.add_text("config", config)

        model = str(self)
        model = model.split('\n')
        model = ['```'] + model + ['```']
        model = '\n'.join(model)
        tb.add_text("model", model)

