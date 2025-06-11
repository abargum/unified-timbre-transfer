import os
import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
import soundfile as sf
import shutil
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../raveish')))
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

import cached_conv as cc
import gin
import nn_tilde
import torch.nn.functional as F
from torchaudio.transforms import Resample
from absl import flags, app
import cdpam
import random
from frechet_audio_distance import FrechetAudioDistance
from pitch_eval_utils import PitchTracker
import matplotlib.pyplot as plt
from raveish.model import RAVE

from core import AudioDistanceV1, MultiScaleSTFT, extract_pitch, search_for_run
import torchcrepe

basic_pitch_module = PitchTracker()

def set_seed(seed):
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Set seed for CPU
    torch.cuda.manual_seed(seed)  # Set seed for CUDA
    torch.cuda.manual_seed_all(seed)  # Set seed for all CUDA devices
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior on GPU
    torch.backends.cudnn.benchmark = False  # Disable optimizations for non-deterministic algorithms
    print(f"Random seed {seed} has been set for reproducibility.")

def calculate_f0_deviation(original, reconstruction, sample_rate, file):
    """Calculate average absolute deviation in fundamental frequency (∆F in Hz)."""
    device = original.device
    
    f0_original = extract_pitch(original.squeeze(1), sr=sample_rate, block_size=2048)
    f0_recon = extract_pitch(reconstruction.squeeze(1), sr=sample_rate, block_size=2048)
    
    min_length = min(f0_original.shape[1], f0_recon.shape[1])
    f0_original = f0_original[:, :min_length]
    f0_recon = f0_recon[:, :min_length]

    f0_original_np = f0_original.squeeze().detach().cpu().numpy()
    f0_recon_np = f0_recon.squeeze().detach().cpu().numpy()
    
    voiced_mask = (f0_original > 0).float()
    abs_deviation = torch.abs(f0_original - f0_recon)
    avg_deviation = torch.sum(abs_deviation * voiced_mask, dim=1) / (torch.sum(voiced_mask, dim=1) + 1e-8)
    
    return avg_deviation

def calculate_loudness_deviation(original, reconstruction):
    """Calculate average absolute deviation in loudness (∆L in dB)."""
    device = original.device
    
    # Convert to dB scale (log amplitude)
    def amplitude_to_db(x):
        return 20 * torch.log10(torch.clamp(torch.abs(x), min=1e-8))
    
    frame_size = 2048
    hop_size = 512
    
    min_length = min(original.shape[-1], reconstruction.shape[-1])
    original = original[..., :min_length]
    reconstruction = reconstruction[..., :min_length]
    
    def get_rms_frames(audio):
        pad_length = (frame_size - (audio.shape[-1] % frame_size)) % frame_size
        padded = F.pad(audio, (0, pad_length))
        frames = padded.unfold(-1, frame_size, hop_size)
        rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-8)
        return amplitude_to_db(rms)
    
    db_original = get_rms_frames(original)
    db_recon = get_rms_frames(reconstruction)
    
    abs_deviation = torch.abs(db_original - db_recon)
    avg_deviation = torch.mean(abs_deviation, dim=-1)
    
    return avg_deviation

def get_jnd_loss(x, y, sample_rate):
    """Calculate Just-Noticeable-Difference loss using CDPAM."""
    device = x.device
    jnd_loss_fn = cdpam.CDPAM(dev=device)
    
    with torch.no_grad():
        resampler = Resample(sample_rate, 22050, dtype=y.dtype).to(device)
        jnd_loss = jnd_loss_fn.forward(resampler(x.squeeze(1)) * 32768, resampler(y.squeeze(1)) * 32768)
    
    return jnd_loss

def adjust_audio_length(y, sr, min_power=14, mode="truncate"):
    """
    Adjust audio length to a power of 2.
    
    Parameters:
        y (numpy array): Audio signal.
        sr (int): Sample rate.
        min_power (int): Minimum power of 2 for length.
        mode (str): "truncate" to cut, "pad" to extend.
    
    Returns:
        numpy array: Adjusted audio signal.
    """
    current_length = len(y)
    min_length = 2 ** min_power

    # Find the nearest power of 2
    target_length = 2 ** int(np.ceil(np.log2(current_length))) 
    lower_power = 2 ** int(np.floor(np.log2(current_length)))

    if mode == "truncate":
        # Truncate to the largest power of 2 that is ≤ current length
        target_length = max(lower_power, min_length)
        y_adj = y[:target_length]

    elif mode == "pad":
        # Pad to the smallest power of 2 that is ≥ current length
        target_length = max(target_length, min_length)
        y_adj = np.pad(y, (0, target_length - current_length), mode='constant')

    else:
        raise ValueError("Invalid mode. Choose 'truncate' or 'pad'.")

    return y_adj

def process_audio_directory(base_dir, model, inst, sample_rate, fad_sample_rate=48000):
    """Process all audio files in directory and calculate metrics."""

    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    multi_scale_stft = lambda: MultiScaleSTFT(scales=[2048, 1024, 512, 256, 128], sample_rate=sample_rate, magnitude=True)
    mstft_loss_fn = AudioDistanceV1(multiscale_stft=multi_scale_stft, log_epsilon=1e-7)

    for module in mstft_loss_fn.modules():
        if hasattr(module, 'window'):
            module.window = module.window.to(device)
    
    metrics = {
        'mstft_loss': [],
        'jnd_loss': [],
        'f0_dev': [],
        'volume_dev': [],
        'mel_dpd': [],
        'mel_jd': []
    }
    
    if sample_rate != fad_sample_rate:
        fad_resampler = Resample(sample_rate, fad_sample_rate).to(device)
    
    valid_files = 0
    
    # Process each audio file
    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(('.wav', '.flac')):
                continue
                
            file_path = os.path.join(root, file)
            print(f"Processing {file_path}...")
            
            instrument = file_path.split('/')[1]
            
            try:
                # Load and process audio
                audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
                audio = adjust_audio_length(audio, sample_rate)
                audio_tensor = torch.tensor(audio, device=device).reshape(1, 1, -1)
                
                with torch.no_grad():
                    out = model(audio_tensor, inst_id=[str(instrument)])
                reconstruction = out[0]
                audio_tensor = audio_tensor.to(device)
                reconstruction = reconstruction.to(device)

                # Calculate metrics
                melody = basic_pitch_module.cals_pitch_metric(audio_tensor, reconstruction, metric='both')
                melody_dpd, melody_jd = melody[0]

                mstft_loss = mstft_loss_fn(audio_tensor, reconstruction)['spectral_distance'].item()

                try:
                    jnd_loss = get_jnd_loss(audio_tensor, reconstruction, sample_rate).item()
                except RuntimeError as e:
                    print(f"Error in JND calculation: {e}")
                    print(f"Input device: {audio_tensor.device}, Reconstruction device: {reconstruction.device}")
                    jnd_loss = get_jnd_loss(audio_tensor.cpu(), reconstruction.cpu(), sample_rate).item()

                f0_loss = calculate_f0_deviation(audio_tensor, reconstruction, sample_rate, file).item()
                vol_loss = calculate_loudness_deviation(audio_tensor, reconstruction).item()
                
                metrics['mstft_loss'].append(mstft_loss)
                metrics['jnd_loss'].append(jnd_loss)
                metrics['f0_dev'].append(f0_loss)
                metrics['volume_dev'].append(vol_loss)
                metrics['mel_dpd'].append(melody_dpd)
                metrics['mel_jd'].append(melody_jd)
                
                print(f"File: {file}")
                print(f"  MSTFT Loss: {mstft_loss:.4f}")
                print(f"  JND Loss: {jnd_loss:.4f}")
                print(f"  F0 Dev: {f0_loss:.4f}")
                print(f"  Volume Dev: {vol_loss:.4f}")
                print(f"  Melody DPD: {melody_dpd:.8f}")
                print(f"  Melody JD: {melody_jd:.8f}")

                if sample_rate != fad_sample_rate:
                    orig_audio_fad = fad_resampler(audio_tensor)
                    recon_audio_fad = fad_resampler(reconstruction)
                else:
                    orig_audio_fad = audio_tensor
                    recon_audio_fad = reconstruction
                
                base_filename = os.path.splitext(os.path.basename(file))[0]
                
                valid_files += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
    
    if valid_files > 0:
        mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
        
        print("\n" + "="*50)
        print(f"Processed {valid_files} files")
        print("Mean Metrics:")
        for metric_name, value in mean_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print("="*50)
        
        return mean_metrics
    else:
        print("No valid files processed.")
        return None

if __name__ == "__main__":

    set_seed(42)

    run = "pretrained/non-causal"
    ema_weights = True
    cc.use_cached_conv(False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    gin.parse_config_file(os.path.join(run, "config.gin"))
    checkpoint = search_for_run(run)
    print("loading checkpoint:", checkpoint)
    
    pretrained = RAVE().to(device)
        
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=device)
        if ema_weights is True and "EMA" in checkpoint["callbacks"]:
            pretrained.load_state_dict(
                checkpoint["callbacks"]["EMA"],
                strict=False)
        else:
            pretrained.load_state_dict(
                checkpoint["state_dict"],
                strict=False)
    else:
        print("No checkpoint found, RAVE will remain randomly initialized")

    pretrained.eval()

    #we need to load the pretrained pitch encoder
    pitch_enc = pretrained.pitch_encoder
    pitch_enc.load_state_dict(torch.load(f"raveish/utils/noncaus2048_mb6.pth", weights_only=True))

    # Test run
    x = torch.rand(1, 1, 2**14, device=device)
    inst_id = ['Trumpet']
    pretrained(x, inst_id=inst_id)
        
    # Remove weight normalization
    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)
    
    # Process audio directory
    base_directory = "test-set"
    sample_rate = 44100
    fad_sample_rate = 16000  # The sample rate required by FAD
    
    metrics = process_audio_directory(
        base_directory, 
        pretrained, 
        inst_id, 
        sample_rate,
        fad_sample_rate
    )
    
    if metrics:
        with open('reconstruction_metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)
        print("Metrics saved to reconstruction_metrics.pkl")