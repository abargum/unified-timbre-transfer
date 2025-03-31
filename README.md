# Unified Timbre Transfer

This repository extends the [RAVE pipeline](https://github.com/acids-ircam/RAVE) with features from [Variational Timbre](https://github.com/acids-ircam/variational-timbre) and [PENN](https://github.com/interactiveaudiolab/penn/tree/master). Key additions:

- Harmonic sine + noise excitation
- Extended decoder with double FiLM conditioning
- Pitch-loss and F0/aperiodicity extraction
- Instrument-dependent harmonic estimation

## Installation

### 1. Install RAVE

```bash
pip install acids-rave
```

*Note:* Install `torch` and `torchaudio` first from [PyTorch](http://www.pytorch.org) to ensure compatibility.

### 2. Install Dependencies

```bash
conda install ffmpeg
pip install torch-pitch-shift torchfcpe mpl-tools julius wandb
```

## Dataset Preparation

1. **Organize your dataset** into instrument folders:

```bash
unified-timbre-transfer/
│── data/
│   ├── Instrument_1/
│       ├── audio1.wav
│       ├── audio2.wav
│   ├── Instrument_n/
│       ├── audio1.wav
│       ├── audio2.wav
```

2. **Modify instrument classes** in `rave/utils/perceptive.py` and `rave/model.py` to match your dataset.
3. **Preprocess data:**
   - Remove silence and split into training/testing sets:
   ```bash
   python data-utils/remove_silence_and_chunk.py --base_dir /path/to/audio --num_samples X
   ```
   - Extract audio features for normalization:
   ```bash
   python rave/extract_features.py --base_dir /path/to/audio
   ```
   - Preprocess audio:
   ```bash
   python scripts/preprocess --input_path /audio/folder/ --output_path /dataset/path/ --channels 1 --lazy
   ```

## Training

Train a causal model with:

```bash
python train.py --config rave/configs/base_config.gin \
                --config rave/configs/causal.gin \
                --db_path_train /path/to/preprocessed-train-data \
                --db_path_test /path/to/preprocessed-test-data \
                --out_path runs --name "experiment_name" --channels 1 --gpu 0
```

### Key Training Parameters (base-config.gin):

- **with_augmentation**: Enable data augmentation
- **with_pitch_loss**: Train pitch encoder
- **load_pitch_enc**: Load pre-trained pitch encoder
- **streaming**: Enable streaming (set `True` after training for causal models)

## Inference

For inference run the script in `inference.ipynb`. Download pre-trained models from: **[link placeholder]**.

## Evaluation

Evaluation results similar to those in the paper can be produced using the scripts in the [evaluation](evaluation/) folder.

- First install basic-pitch:
```bash
cd evalution 
python data-utils/remove_silence_and_chunk.py --base_dir /path/to/audio --num_samples X
```

- Then install remaining dependencies:
```bash
pip install cdpam
pip install frechet_audio_distance
```

Run the desired evaluation script:
```bash
python evaluation/ablation_metrics_rec.py
```
