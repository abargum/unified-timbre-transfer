import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rave')))

from core import a_weighted_loudness, extract_pitch

import os
import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
import argparse

def get_features(file_path, sr):
    x, sr = librosa.load(file_path, sr=sr)
    x = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(torch.device('cuda'))
    loudness = a_weighted_loudness(x, sr, n_fft=2048, block_size=2048, to_linear=False)
    pitch = extract_pitch(x.squeeze(0), sr, 2048)
    pitch = pitch[~torch.isnan(pitch)]
    pitch = pitch[pitch > 0]
    return loudness.cpu().numpy().flatten(), pitch.cpu().numpy().flatten()

def process_audio_directory(base_dir, output_path, sample_rate):
    instrument_data = {}
    
    for root, dirs, files in os.walk(base_dir):
        if root == base_dir:
            continue
        
        rel_path = os.path.relpath(root, base_dir)
        path_parts = rel_path.split(os.sep)
        
        has_audio_files = any(file.endswith(('.wav', '.flac')) for file in files)
        if not has_audio_files:
            continue
            
        instrument = path_parts[0]
        
        if instrument not in instrument_data:
            instrument_data[instrument] = {
                'all_loudness': [],
                'all_pitch': [],
                'valid_files': 0
            }
        
        for file in files:
            if not file.endswith(('.wav', '.flac')):
                continue
            file_path = os.path.join(root, file)
            print(f"Processing {file_path}...")
            
            try:
                loudness, pitch = get_features(file_path, sample_rate)
                instrument_data[instrument]['all_loudness'].append(loudness)
                instrument_data[instrument]['all_pitch'].append(pitch)
                instrument_data[instrument]['valid_files'] += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    final_data = {}
    for instrument, data in instrument_data.items():
        if data['valid_files'] > 0:
            all_loudness = np.concatenate(data['all_loudness'])
            all_pitch = np.concatenate(data['all_pitch'])
            
            mean_loudness = np.mean(all_loudness)
            std_loudness = np.std(all_loudness)
            mean_pitch = np.mean(all_pitch)
            
            print(f'{instrument}, mean: {mean_loudness}, std: {std_loudness}, f0: {mean_pitch}')
            final_data[instrument] = {
                'mean': mean_loudness,
                'std': std_loudness,
                'mean_f0': mean_pitch,
                'n_files': data['valid_files']
            }
        else:
            print(f"No valid audio files found for {instrument}.")
    
    with open(output_path, 'wb') as f:
        pickle.dump(final_data, f)
    print(f"Saved loudness statistics to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files and extract loudness/pitch features.")
    parser.add_argument("--base_dir", type=str, help="Path to the base directory containing audio files.")
    parser.add_argument("--output_file", type=str, default="src/utils/loudness_stats.pkl", help="Path to save the extracted statistics (pickle file).")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate for processing audio files (default: 44100).")

    args = parser.parse_args()

    process_audio_directory(args.base_dir, args.output_file, args.sample_rate)
