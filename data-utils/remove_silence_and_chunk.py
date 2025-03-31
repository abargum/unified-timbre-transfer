import os
import librosa
import numpy as np
import torch
import random
import shutil
import argparse
from scipy.io.wavfile import read, write
import subprocess
import tqdm

def detect_silence(path, time, noise_threshold=-23):
    command = [
        "ffmpeg", "-i", path,
        "-af", f"silencedetect=n={noise_threshold}dB:d={time}",
        "-f", "null", "-"
    ]
    
    out = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = out.communicate()
    s = stderr.decode("utf-8")    
    k = s.split('[silencedetect @')
    print("Number of Silence Sections Found:", len(k) - 1)
    
    if len(k) == 1:
        return None
    
    start, end = [], []
    
    for i in range(1, len(k)):
        x = k[i].split(']')[1]
        if 'silence_end' in x:
            x = x.split('|')[0].split(':')[1].strip()
            end.append(float(x))
        elif 'silence_start' in x:
            x = x.split(':')[1].split('size')[0].replace('\r', '').replace('\n', '').strip()
            if '[' in x:
               x = x.split('[')[0]
            start.append(float(x))
    
    length = min(len(start), len(end))
    return list(zip(start[:length], end[:length]))

def get_audio_duration(file_path):
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
               "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def split_audio(file, sil, out_dir, min_duration=1.0, max_segment_length=30.0):
    rate, aud = read(file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    non_sil = []
    tmp = 0
    ed = len(aud) / rate  
    if sil:
        for i in range(len(sil)):
            if sil[i][0] > tmp:
                non_sil.append((tmp, sil[i][0]))
            tmp = sil[i][1]
        if sil[-1][1] < ed:
            non_sil.append((sil[-1][1], ed))
    else:
        non_sil.append((0, ed))
    
    final_segments = []
    for start, end in non_sil:
        duration = end - start
        if duration > max_segment_length:
            num_segments = int(np.ceil(duration / max_segment_length))
            segment_length = duration / num_segments
            for i in range(num_segments):
                seg_start = start + i * segment_length
                seg_end = min(seg_start + segment_length, end)
                final_segments.append((seg_start, seg_end))
        else:
            final_segments.append((start, end))
    
    saved_chunks = []
    for idx, (start, end) in enumerate(final_segments):
        duration = end - start
        if duration >= min_duration:
            chunk = aud[int(start * rate): int(end * rate)]
            if np.any(np.abs(chunk) > 100):
                out_path = os.path.join(out_dir, f"chunk_{idx}.wav")
                write(out_path, rate, np.array(chunk))
                print(f"Saved: {out_path} (Duration: {duration:.2f}s)")
                saved_chunks.append(out_path)
    return saved_chunks, final_segments

def create_test_set(chunks_dir, test_dir, num_samples=2, min_length=3.0, max_length=10.0):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    for instrument in os.listdir(chunks_dir):
        instrument_dir = os.path.join(chunks_dir, instrument)
        if os.path.isdir(instrument_dir):
            instrument_test_dir = os.path.join(test_dir, instrument)
            audio_dir = os.path.join(instrument_test_dir, "audio")
            if not os.path.exists(audio_dir):
                os.makedirs(audio_dir)
            valid_chunks = []
            for chunk_dir in os.listdir(instrument_dir):
                if "_chunks" in chunk_dir:
                    chunk_path = os.path.join(instrument_dir, chunk_dir)
                    if os.path.isdir(chunk_path):
                        for chunk in os.listdir(chunk_path):
                            if chunk.endswith(('.wav', '.flac')):
                                full_path = os.path.join(chunk_path, chunk)
                                duration = get_audio_duration(full_path)
                                if min_length <= duration <= max_length:
                                    valid_chunks.append(full_path)
            if valid_chunks:
                samples = random.sample(valid_chunks, min(num_samples, len(valid_chunks)))
                for i, sample in enumerate(samples):
                    sample_name = f"{instrument}_sample_{i+1}.wav"
                    dest_path = os.path.join(audio_dir, sample_name)
                    shutil.copy(sample, dest_path)
                    print(f"Added to test set: {dest_path}")

def process_audio_directory(base_dir, output_directory, test_directory=None, num_samples=2,
                           silence_threshold=-23, silence_duration=4.0,
                           min_chunk_duration=2.0, max_chunk_duration=30.0):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for root, dirs, files in os.walk(base_dir):
        if root == base_dir:
            continue
        
        instrument = os.path.basename(root)
        instrument_dir = os.path.join(output_directory, instrument)
        
        if not os.path.exists(instrument_dir):
            os.makedirs(instrument_dir)
        
        for file in files:
            if file.endswith(('.wav', '.flac')):
                file_path = os.path.join(root, file)
                output_folder = os.path.join(instrument_dir, file.replace('.wav', '').replace('.flac', '') + '_chunks')
                print(f"\nProcessing {file_path}...")
                silence_list = detect_silence(file_path, silence_duration, silence_threshold)
                if silence_list or True:
                    split_audio(file_path, silence_list, output_folder, 
                              min_duration=min_chunk_duration, 
                              max_segment_length=max_chunk_duration)
    if test_directory:
        print("\nCreating test set...")
        create_test_set(output_directory, test_directory, num_samples=num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files and create a dataset.")
    parser.add_argument("--base_dir", type=str, help="Path to the base directory containing audio files.")
    parser.add_argument("--training_dir", type=str, default="train-set", help="Path to the training dataset directory.")
    parser.add_argument("--test_dir", type=str, default="test-set", help="Path to the test dataset output directory.")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples per instrument to include in test set.")
    parser.add_argument("--silence_threshold", type=float, default=-23, help="Silence detection threshold in dB (default: -23).")
    parser.add_argument("--silence_duration", type=float, default=4.0, help="Minimum silence duration in seconds (default: 4.0).")
    parser.add_argument("--min_chunk_duration", type=float, default=2.0, help="Minimum chunk duration in seconds (default: 2.0).")
    parser.add_argument("--max_chunk_duration", type=float, default=30.0, help="Maximum chunk duration in seconds (default: 30.0).")
    args = parser.parse_args()
    process_audio_directory(args.base_dir, args.training_dir, args.test_dir, args.silence_threshold, args.silence_duration, args.min_chunk_duration, args.max_chunk_duration)
