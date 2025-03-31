# Unified Timbre Transfer

This repository builds on top of the [RAVE pipeline](https://github.com/acids-ircam/RAVE) and a few scripts of the [Variational Timbre](https://github.com/acids-ircam/variational-timbre) and the [PENN](https://github.com/interactiveaudiolab/penn/tree/master) repositories. Main additions include:

- Harmonic sine + noise excitation
- Extended decoder with double FiLM conditioning
- Pitch-loss and training for F0 and aperiodicity extraction
- Instrument-dependent harmonic estimation

## Installation

First, install RAVE:

```bash
pip install acids-rave
```

**Warning** It is strongly advised to install `torch` and `torchaudio` before `acids-rave`, so you can choose the appropriate version of torch on the [library website](http://www.pytorch.org). For future compatibility with new devices (and modern Python environments), `rave-acids` does not enforce torch==1.13 anymore.

You will need **ffmpeg** on your computer as well as a few other libraries:

```bash
conda install ffmpeg
pip install torch-pitch-shift
pip install torchfcpe
pip install mpl-tools
pip install julius
pip install wandb
```

### Dataset preparation

Download your desired dataset - in order to keep track of the instrument specific audio belongs to, the data should be sorted into instrument folders i.e:

```bash
unified-timbre-transfer/
│── data/
│   ├── Instrument1
│   ├────├ audio1.wav
│   ├────├ audio2.wav
```

Remove any silence from your dataset and split it into training and test sets:

```bash
python data-utils/remove_silence_and_chunk.py --base_dir /path/to/audio
```

Hereafter extract audio features for normalization:

```bash
python rave/extract_features.py --base_dir /path/to/audio
```

Now preprocess the audio. You will need to preprocess both the training and test folders created above:

```bash
python scripts.preprocess --input_path /audio/folder/ --output_path /dataset/path/ --channels 1 --lazy
```

### Training

To train a causal model, run the configuration (remove the config keyword for causal.gin to train a non-causal model).

```bash
python train.py --config rave/configs/base_config.gin --config rave/configs/causal.gin --db_path_train path/to/preprocessed-train-data --db_path_test path/to/preprocessed-test-data --out_path runs --name "name" --channels 1 --gpu 0
```

### Inference

We provide an inference script in inference.ipynb. In order to load the pre-trained models, please download it from: xxx