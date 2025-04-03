import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cached_conv as cc
import gin
import soundfile as sf
import tqdm
import subprocess
import re
import random
import fnmatch
import shutil
from frechet_audio_distance import FrechetAudioDistance

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rave')))
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from rave.model import RAVE

from core import extract_pitch, search_for_run

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


def load_model(run_path, device):
    """Load and prepare the RAVE model."""
    cc.use_cached_conv(False)
    
    gin.parse_config_file(os.path.join(run_path, "config.gin"))
    checkpoint = search_for_run(run_path)
    print("Loading checkpoint:", checkpoint)
    
    model = RAVE().to(device)
        
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=device)
        if "EMA" in checkpoint["callbacks"]:
            model.load_state_dict(
                checkpoint["callbacks"]["EMA"],
                strict=False)
        else:
            model.load_state_dict(
                checkpoint["state_dict"],
                strict=False)
    else:
        print("No checkpoint found, RAVE will remain randomly initialized")

    model.eval()

    # Load pitch encoder
    pitch_enc = model.pitch_encoder
    pitch_enc.load_state_dict(torch.load(f"rave/utils/noncaus2048_mb6.pth", weights_only=True))

    # Test run
    x = torch.rand(1, 1, 2**14, device=device)
    inst_id = ['Trumpet']
    model(x, inst_id=inst_id)
        
    # Remove weight normalization
    for m in model.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)
    
    return model


def get_instrument_files(input_folder):
    """
    Scan the input folder structure to find all instrument files.
    Returns a dictionary mapping instrument names to lists of audio file paths.
    """
    instrument_files = {}
    
    for instrument in os.listdir(input_folder):
        instrument_path = os.path.join(input_folder, instrument)
        if not os.path.isdir(instrument_path):
            continue
            
        audio_path = os.path.join(instrument_path, "audio")
        if not os.path.isdir(audio_path):
            continue
            
        files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith(".wav")]
        if files:
            instrument_files[instrument] = files
    
    return instrument_files


def collect_all_audio_files(folder):
    """
    Collect all audio files from a folder and its subfolders.
    Returns a list of all .wav file paths.
    """
    audio_files = []
    
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    return audio_files


def prepare_fad_folder(files, output_folder):
    """
    Prepare a folder with all audio files for FAD evaluation.
    Copies all files to a flat structure for easier FAD calculation.
    
    Args:
        files: List of audio file paths
        output_folder: Folder to copy files to
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for i, file_path in enumerate(files):
        # Create a unique filename to avoid collisions
        filename = f"audio_{i:04d}.wav"
        output_path = os.path.join(output_folder, filename)
        
        # Copy the file (using librosa to ensure consistent format)
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        sf.write(output_path, audio, sr)
    
    return output_folder


def process_timbre_transfer(input_folder, output_folder, model, device):
    """
    Process each input instrument file and transfer it to all target instruments.
    
    Args:
        input_folder: Path to the input folder with instrument/audio structure
        output_folder: Path to save transferred audio files
        model: The loaded RAVE model
        device: Torch device
    """
    # Get all instrument files
    instrument_files = get_instrument_files(input_folder)
    instruments = list(instrument_files.keys())
    
    # Create output folders for each target instrument
    for instrument in instruments:
        os.makedirs(os.path.join(output_folder, instrument), exist_ok=True)
    
    # Create metrics dictionaries to store results
    f0_deviations = {}
    loudness_deviations = {}
    
    # Process each source file for each target instrument
    for source_instrument, source_files in tqdm.tqdm(instrument_files.items(), desc="Processing source instruments"):
        for source_file in tqdm.tqdm(source_files, desc=f"Files from {source_instrument}"):
            filename = os.path.basename(source_file)
            
            # Load source audio
            audio, sr = librosa.load(source_file, sr=44100, mono=True)
            audio = adjust_audio_length(audio, sr)
            audio_tensor = torch.tensor(audio, device=device).reshape(1, 1, -1)
            
            # Transfer to each target instrument
            for target_instrument in instruments:
                # Create output filename
                base_filename = os.path.splitext(filename)[0]
                output_filename = f"{base_filename}_{source_instrument}_to_{target_instrument}.wav"
                output_path = os.path.join(output_folder, target_instrument, output_filename)
                
                # Skip if already processed
                if os.path.exists(output_path):
                    continue
                
                # Perform timbre transfer
                with torch.no_grad():
                    transferred = model(audio_tensor, inst_id=[target_instrument])
                    transferred = transferred[0]
                
                # Calculate F0 and loudness deviations
                key = f"{source_instrument}_{target_instrument}"
                file_key = f"{base_filename}_{key}"
                
                f0_dev = calculate_f0_deviation(audio_tensor, transferred, 44100, file_key)
                loudness_dev = calculate_loudness_deviation(audio_tensor, transferred)
                
                # Store deviation metrics
                if key not in f0_deviations:
                    f0_deviations[key] = []
                    loudness_deviations[key] = []
                
                f0_deviations[key].append(f0_dev.item())
                loudness_deviations[key].append(loudness_dev.item())
                
                # Save transferred audio
                audio_out = transferred.squeeze().detach().cpu().numpy()
                sf.write(output_path, audio_out, 44100)
                
                print(f"Transferred: {source_instrument}/{filename} → {target_instrument}")
    
    # Calculate average metrics
    avg_f0_deviations = {key: sum(values) / len(values) for key, values in f0_deviations.items()}
    avg_loudness_deviations = {key: sum(values) / len(values) for key, values in loudness_deviations.items()}
    
    # Calculate overall averages
    all_f0_values = [item for sublist in f0_deviations.values() for item in sublist]
    all_loudness_values = [item for sublist in loudness_deviations.values() for item in sublist]
    overall_f0_avg = sum(all_f0_values) / len(all_f0_values) if all_f0_values else 0
    overall_loudness_avg = sum(all_loudness_values) / len(all_loudness_values) if all_loudness_values else 0
    
    # Save metrics to a file
    metrics_path = os.path.join(output_folder, "metrics_results.txt")
    with open(metrics_path, "w") as f:
        f.write("F0 and Loudness Deviation Metrics\n")
        f.write("================================\n\n")
        
        f.write("F0 Deviations (Hz) - Lower is better\n")
        f.write("--------------------------\n")
        for key, value in avg_f0_deviations.items():
            source, target = key.split('_')
            f.write(f"{source} → {target}: {value:.2f} Hz\n")
        f.write(f"\nOverall Average F0 Deviation: {overall_f0_avg:.2f} Hz\n\n")
        
        f.write("Loudness Deviations (dB) - Lower is better\n")
        f.write("--------------------------\n")
        for key, value in avg_loudness_deviations.items():
            source, target = key.split('_')
            f.write(f"{source} → {target}: {value:.2f} dB\n")
        f.write(f"\nOverall Average Loudness Deviation: {overall_loudness_avg:.2f} dB\n")
    
    print(f"F0 and Loudness metrics saved to {metrics_path}")
    
    return avg_f0_deviations, avg_loudness_deviations, overall_f0_avg, overall_loudness_avg


def calculate_fad(original_dir, recon_dir, model_name="vggish", sample_rate=16000):
    """Calculate Fréchet Audio Distance between two directories."""
    print(f"Calculating FAD between {original_dir} and {recon_dir}...")
    
    frechet = FrechetAudioDistance(
        model_name=model_name,
        sample_rate=sample_rate,
        use_pca=False, 
        use_activation=False,
        verbose=False
    )
    
    fad_score = frechet.score(
        original_dir, 
        recon_dir, 
        dtype="float32"
    )
    
    print(f"FAD Score: {fad_score:.4f}")
    return fad_score


def calculate_mmd(training_dir, output_dir):
    """
    Calculate MMD between training data and reconstructed transfers for each instrument.
    Uses the nas-eval command line tool.
    
    Args:
        training_dir: Directory containing training data with instrument subfolders
        output_dir: Directory containing output files with instrument subfolders
    
    Returns:
        Dictionary of MMD scores per instrument and the mean MMD score
    """
    mmd_scores = {}
    
    # Get list of instruments in output directory
    instruments = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    for instrument in instruments:
        training_instrument_dir = os.path.join(training_dir, instrument)
        output_instrument_dir = os.path.join(output_dir, instrument)
        
        # Skip if training data for this instrument doesn't exist
        if not os.path.isdir(training_instrument_dir):
            print(f"Warning: No training data found for {instrument}, skipping MMD calculation")
            continue
        
        print(f"Calculating MMD for {instrument}...")
        try:
            result = subprocess.run(
                ['nas-eval', 'timbre', training_instrument_dir, output_instrument_dir],
                capture_output=True, 
                text=True
            )
            
            # Extract MMD value from output
            matrix_match = re.search(r'MMD matrix: \n\[\[([\d\.]+)\]\]', result.stderr)
            if matrix_match:
                mmd_value = float(matrix_match.group(1))
                mmd_scores[instrument] = mmd_value
                print(f"MMD for {instrument}: {mmd_value:.4f}")
            else:
                print(f"Warning: Could not extract MMD value for {instrument}")
                print(f"Command output: {result.stderr}")
        except Exception as e:
            print(f"Error calculating MMD for {instrument}: {e}")
    
    # Calculate mean MMD score
    if mmd_scores:
        mean_mmd = sum(mmd_scores.values()) / len(mmd_scores)
        mmd_scores['mean'] = mean_mmd
        print(f"Mean MMD across all instruments: {mean_mmd:.4f}")
    else:
        mean_mmd = None
        print("Warning: No MMD scores calculated")
    
    return mmd_scores


if __name__ == "__main__":

    set_seed(42)
    
    # Configuration
    run_path = "pretrained/non-causal"
    
    input_folder = "test-set"  # Folder with instrument/audio/*.wav structure
    output_folder = "timbre_transfers"  # Folder to save transferred audio
    training_data_folder = "urmp-data"  # Folder/background set containing training data for FAD and MMD calculation
    
    fad_output_folder = "fad_output"  # Temporary folder for FAD evaluation
    fad_input_folder = "fad_input"    # Temporary folder for FAD evaluation
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        
    os.makedirs(output_folder)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(run_path, device)
    
    # Process timbre transfer - now returning metrics too
    f0_deviations, loudness_deviations, overall_f0_avg, overall_loudness_avg = process_timbre_transfer(
        input_folder, output_folder, model, device
    )
    
    # Calculate FAD score
    print("\n=== Calculating FAD Score ===")
    
    # Prepare folders for FAD evaluation
    print("Preparing audio files for FAD evaluation...")
    
    # Collect all audio files
    input_audio_files = collect_all_audio_files(training_data_folder)
    output_audio_files = collect_all_audio_files(output_folder)
    
    # Prepare FAD folders
    fad_input_dir = prepare_fad_folder(input_audio_files, fad_input_folder)
    fad_output_dir = prepare_fad_folder(output_audio_files, fad_output_folder)
    
    # Calculate overall FAD on all audio
    fad_score = calculate_fad(fad_input_dir, fad_output_dir)
    
    # Calculate MMD scores
    print("\n=== Calculating MMD Scores ===")
    mmd_scores = calculate_mmd(training_data_folder, output_folder)
    
    # Save evaluation results
    with open(os.path.join(output_folder, "evaluation_results.txt"), "w") as f:
        f.write("Timbre Transfer Evaluation Results\n")
        f.write("================================\n\n")
        
        f.write("F0 Deviation (Hz) - Lower is better\n")
        f.write("--------------------------\n")
        f.write(f"Overall average F0 deviation: {overall_f0_avg:.2f} Hz\n\n")
        
        f.write("Loudness Deviation (dB) - Lower is better\n")
        f.write("--------------------------\n")
        f.write(f"Overall average loudness deviation: {overall_loudness_avg:.2f} dB\n\n")
        
        f.write("FAD Score (Lower is better)\n")
        f.write("--------------------------\n")
        f.write(f"Overall FAD score: {fad_score:.4f}\n\n")
        
        f.write("MMD Scores (Lower is better)\n")
        f.write("--------------------------\n")
        if mmd_scores:
            for instrument, score in mmd_scores.items():
                if instrument != 'mean':
                    f.write(f"MMD for {instrument}: {score:.4f}\n")
            f.write(f"\nMean MMD across all instruments: {mmd_scores.get('mean', 'N/A'):.4f}\n")
        else:
            f.write("No MMD scores calculated\n")
    
    print("\nEvaluation complete. Results saved to evaluation_results.txt")
    
    # Clean up temporary FAD folders (optional)
    shutil.rmtree(fad_input_folder)
    shutil.rmtree(fad_output_folder)