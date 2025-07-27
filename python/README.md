# Sound Matching an Analogue Levelling Amplifier Using the Newton-Raphson Method

The experimental code is adapted from https://github.com/DiffAPF/LA-2A.


## Getting started

Please install the required packages, including the differentiable compressor [torchcomp](https://github.com/yoyololicon/torchcomp), by running:

```bash
pip install -r requirements.txt
```

## Training 4A2A compressor

Firstly, you need to download the SignalTrain dataset from [here](https://zenodo.org/records/3824876).
The following code assumes that `$SIGNALTRAIN` is the path to the dataset.

In the paper, we fit the differentiable feed-forward compressor starting with the maximum peak reduction of 100.
The training configurations and initial parameters are listed in `cfg/config.yaml`.
You can modify the configurations directly or use command-line arguments to override them (for details, please refer to the [Hydra documentation](https://hydra.cc/docs/intro/)).

For peak reduction of 100 in compressor mode, you should run the following command:

```bash
python train_comp.py ckpt_dir=$CHECKPOINTPATH/la2a_100 data.train.input=$SIGNALTRAIN/Train/input_158_.wav data.train.target=$SIGNALTRAIN/Train/target_158_LA2A_3c__0__100.wav
```

After running the command, the learnt parameters and training configs will be saved under the directory `$CHECKPOINTPATH/comp/la2a_100`.
Please use absolute paths for the `ckpt_dir`, `data.train.input`, and `data.train.target` arguments, due to the current limitation of Hydra.
If your GPU does not have enough memory, you can specify the `batch_size` argument to reduce the memory usage, e.g., `data.batch_size=8`.
By default, the script uses the entire sequence, corresponding to `data.batch_size=-1`.
The `epochs` argument specifies the number of training epochs, though if the Newton method converges, the training will stop early.

Next, you should train the compressor with peak reduction of 95 using the previous run parameters as the initial parameters.

```bash
python train_comp.py ckpt_dir=$CHECKPOINTPATH/la2a_95  data.train.input=$SIGNALTRAIN/Train/input_157_.wav data.train.target=$SIGNALTRAIN/Train/target_157_LA2A_3c__0__95.wav  compressor.init_ckpt=$CHECKPOINTPATH/la2a_100/logits.pt
```

This process should be repeated for each peak reduction level you want to train, e.g., 90, 85, ..., down to 40.
To train with limiter mode, select the wave file with `3c__1__` in the name, e.g., `Train/target_179_LA2A_3c__1__100.wav`, and repeat the same process as above starting from the peak reduction of 100.

### Evaluation

After [training](#training-4a2a-compressor), you should have a directory `$CHECKPOINTPATH` containing subdirectories for each peak reduction level, e.g., `la2a_100`, `la2a_95`, ..., `la2a_40`.
The following command will gather the learnt parameters from all the subdirectories, calculate the error signal ratio (ESR) of the compressor, and store the results in a CSV file.
Additionally, the ESR of linear and spline interpolations of the parameters at peak reduction levels of 95, 85, 75, 65, 55, and 45 will also be calculated and stored in the same CSV file.

```bash
python eval.py $CHECKPOINTPATH comp.csv
```

Pre-computed evaluation results are available [here](evaluations/).

## Baselines

### VST plugins (Windows system only)

Please make sure you have the following LA-2A VST3 plugins installed:
- [Cakewalk CA-2A](https://legacy.cakewalk.com/Products/CA-2A)
- [Waves CLA-2A](https://www.waves.com/plugins/cla-2a-compressor-limiter)
- [UAD Audio UAD LA-2A](https://www.uaudio.com/uad-plugins/compressors-limiters/teletronix-la-2a-tube-compressor.html)

The following command will render audio files with `*3c*` substrings and save them in the specified `$OUTPUTPATH`.

#### CA-2A

```bash
python vst_render.py $SIGNALTRAIN $OUTPUTPATH --vst "C:\Program Files\Common Files\VST3\CA2ALevelingAmplifier\CA-2ALevelingAmplifier_64.vst3" --brand cakewalk --gain 0 --out-gain 38
```

The exact path to the CA-2A VST3 plugin may vary depending on your system and installation.
Use `--mode` to specify the mode, e.g., `--mode 1` for limiter mode. Default is compressor mode.

#### CLA-2A

```bash
python vst_render.py $SIGNALTRAIN $OUTPUTPATH --vst "C:\Program Files\Common Files\VST3\WaveShell1-VST3 15.5_x64.vst3" --brand waves --gain -16 --out-gain 50
```
The exact path to the Waves VST3 plugin may vary depending on your system and installation.
Use `--mode` to specify the mode, e.g., `--mode 1` for limiter mode. Default is compressor mode.

#### UAD 

```bash
python vst_render.py $SIGNALTRAIN $OUTPUTPATH --vst "C:\Program Files\Common Files\VST3\uaudio_teletronix_la-2a_tc.vst3\Contents\x86_64-win\uaudio_teletronix_la-2a_tc.vst3" --brand uad --gain -12 --out-gain 46
```
The exact path to the UAD VST3 plugin may vary depending on your system and installation.
Use `--mode` to specify the mode, e.g., `--mode 1` for limiter mode. Default is compressor mode.

### GRU Make-up Gain

Please first run the following command to process the SignalTrain input audio files with the trained compressor, but without the make-up gain.
(To render with make-up gain, please comment out the relevant line in `4a2a_render.py`.)

```bash
python 4a2a_render.py
```

Please modify the path variables that point to the SignalTrain dataset, the output directory, and the checkpoint directory in `4a2a_render.py` before running the command.

Next, run the following command to train the GRU make-up gain model.

```bash
python train_gru.py
```

Please modify the path variables that point to the directory containing the processed audio and the SignalTrain dataset in `train_gru.py` before running the command.
Afterwards, you should have multiple checkpoints with the name `gru_jit_no_overlap_{epoch}_{loss}.pt` in the current directory.
We pick the one with the lowest loss as the final model.

Lastly, run the following command to render the processed audio files with the best GRU make-up gain model.

```bash
python gru_render.py
```

Please modify the path variables, model path, output directory, and the file name pattern (e.g., `3c__1` for limiter mode) in `gru_render.py` before running the command.

#### Real-time Neutone plugin

To convert the GRU make-up gain model into a format compatible with the [Neutone FX](https://neutone.ai/fx), please run the following command:

```bash
python convert.py
```

Please modify the checkpoint path that needs to be converted in `convert.py` before running the command.
It will create a directory `neutone_gru` containing the converted model files, which can be loaded in the Neutone FX plugin.

## Comparison and Visualisation

After rendering the audio files with the trained compressor and the baselines, you can run the [compare](compare.ipynb) notebook to compute the ESR and Loudness Dynamic Range difference ($`\Delta`$LDR) between the rendered audio files and the target audio files.
It also plots Figure 6 in the paper.
Pre-computed evaluation scores are available [here](evaluations/scores.csv). 

## Citation

```bibtex
@inproceedings{ycy2025Newtone,
  title={Sound Matching an Analogue Levelling Amplifier Using the Newton-Raphson Method},
  author={Chin-Yun Yu and György Fazekas},
  booktitle={AES International Conference on Artificial Intelligence and Machine Learning for Audio},
  address={London, UK},
  year={2025},
}
```
