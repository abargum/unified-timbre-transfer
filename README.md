# Unified Timbre Transfer

This repository uses the [RAVE pipeline](https://github.com/acids-ircam/RAVE) as a training baseline and extends it with the contributions from the paper: [Unified Timbre Transfer: A Compact Model for Real-Time Multi-Instrument Sound Morphing]().

It also includes code and inspiration from certain parts of the [Variational Timbre](https://github.com/acids-ircam/variational-timbre) and [PENN](https://github.com/interactiveaudiolab/penn/tree/master) repositories.

Key additions:

- Harmonic sine + noise excitation
- Extended decoder with double FiLM conditioning
- Pitch-loss and F0/aperiodicity extraction
- Instrument-dependent harmonic estimation

## Installation

### 1. Install RAVE and its dependencies

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

1. **Organize your dataset** into instrument folders (you will manually have to sort the URMP dataset to match the targets of the paper):

```bash
unified-timbre-transfer/
│── data/
│   ├── Instrument_1/
│         ├── audio1.wav
│         ├── audio2.wav
│   ...
│
│   ├── Instrument_n/
│         ├── audio1.wav
│         ├── audio2.wav
```

2. **Modify instrument classes** in `raveish/utils/perceptive.py` and `raveish/model.py` to match your dataset or leave as is to follow the procedure from the paper.
3. **Preprocess data:**
   - Remove silence and split into training/testing sets:
   ```bash
   python data-utils/remove_silence_and_chunk.py --base_dir /path/to/audio --num_samples X
   ```
   - Extract audio features for normalization:
   ```bash
   python data-utils/extract_features.py --base_dir /path/to/audio
   ```
   - Preprocess train and test audio:
   ```bash
   python scripts/preprocess.py --input_path train-set/ --output_path train-data/ --channels 1 --lazy
   python scripts/preprocess.py --input_path test-set/ --output_path test-data/ --channels 1 --lazy
   ```

## Training

Train a causal model with:

```bash
python train.py --config raveish/configs/base_config.gin \
                --config raveish/configs/causal.gin \
                --db_path_train /path/to/preprocessed-train-data \
                --db_path_test /path/to/preprocessed-test-data \
                --out_path runs --name "experiment_name" --channels 1 --gpu 0
```

### Key Training Parameters (base-config.gin):

- **with_augmentation**: Enable data augmentation
- **with_pitch_loss**: Train pitch encoder
- **load_pitch_enc**: Load pre-trained pitch encoder
- **streaming**: Enable streaming (set to `True` AFTER training of causal models)

## Inference

For inference and live sound examples, run the cells in `inference.ipynb`. Create and place the pre-trained model folders in "pretrained" or change the path in the script. The pre-trained model weights can be downloaded from: [models](https://drive.google.com/drive/folders/1-JXWJCOnS6bK5ZgBjA6LIErZ8tqMwYjo?usp=drive_link).

## Evaluation

Evaluation results similar to those in the paper can be produced using the scripts in the [evaluation](evaluation/) folder.

- First install basic-pitch:
```bash
cd evalution 
git clone https://github.com/gudgud96/basic-pitch-torch.git
```

- Then install remaining dependencies:
```bash
pip install cdpam
pip install frechet_audio_distance
```

- Run the desired evaluation scripts:
```bash
python evaluation/ablation_metrics_rec.py
python evaluation/ablation_metrics_tt.py
```
