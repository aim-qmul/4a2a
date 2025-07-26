# Sound Matching an Analogue Levelling Amplifier Using the Newton-Raphson Method

The experimental code is adapted from https://github.com/DiffAPF/LA-2A.


## Getting started

Please install the required packages, including the differentiable compressor [torchcomp](https://github.com/yoyololicon/torchcomp), by running:

```bash
pip install -r requirements.txt
```

## Training

Firstly, you need to download the SignalTrain dataset from [here](https://zenodo.org/records/3824876).
The following code will assumes that `$SIGNALTRAIN` is the path to the dataset.

In the paper, we fit the differentiable feed-forward compressor starting with the maximum peak reduction of 100.
The training configurations and initial parameters are listed in `cfg/config.yaml`.
You can modify the configurations directly or use the command line arguments to override them (for details, please refer to the [Hydra documentation](https://hydra.cc/docs/intro/)).

For peak reduction of 100 in compressor mode, you should run the following command:

```bash
python train_comp.py ckpt_dir=$CHECKPOINTPATH/comp_100 data.train.input=$SIGNALTRAIN/Train/input_158_.wav data.train.target=$SIGNALTRAIN/Train/target_158_LA2A_3c__0__100.wav
```

After running the command, the learnt parameters and training configs will be saved under the directory `$CHECKPOINTPATH/comp/comp_100`.
Please use absolute paths for the `ckpt_dir`, `data.train.input`, and `data.train.target` arguments, due to the current limitation of Hydra.
If your GPU does not have enough memory, you can specify the `batch_size` argument to reduce the memory usage, e.g., `data.batch_size=8`.
By default, the training use the whole sequence, which corresponds to `data.batch_size=-1`.
The `epochs` argument specifies the number of training epochs, though if the newton method converges, the training will stop early.

Next, you can should train the compressor with peak reduction of 95 using the previous runned parameters as the initial parameters.

```bash
python train_comp.py ckpt_dir=$CHECKPOINTPATH/comp_95  data.train.input=$SIGNALTRAIN/Train/input_157_.wav data.train.target=$SIGNALTRAIN/Train/target_157_LA2A_3c__0__95.wav  compressor.inits.params=$CHECKPOINTPATH/comp_100/logits.pt
```

This process should be repeated for each peak reduction level you want to train, e.g., 90, 85, ..., down to 40.
To train with limiter mode, simply select the wave file with `3c__1__` in the name, e.g., `Train/target_179_LA2A_3c__1__100.wav`, and repeat the same process as above starting from the peak reduction of 100.

## Evaluation

After [training](#training), you should have a directory `$CHECKPOINTPATH` containing the checkpoints for each peak reduction level.

You can use your checkpoints `ckpt.yaml` or our provided learned parameters to evaluate the compressor.
Given a wave file, you can compress it using the following command:

```bash
python test_comp.py ckpt.yaml input.wav output.wav
```


## Additional notes
- `cfg/data/ff_*.yaml` are configurations for the feed-forward compressor experiments (FF-A/B/C in the paper). Please use `digital_compressor.py` to get the targets if you want to reproduce the experiments.

## Links

- [torchcomp](https://github.com/yoyololicon/torchcomp): Differentiable compressor implementation.
- [training logs](https://wandb.ai/iamycy/torchcomp-la2a/): All training logs of the compressor experiments in the paper.

## Citation

```bibtex
@inproceedings{ycy2024diffapf,
  title={Differentiable All-pole Filters for Time-varying Audio Systems},
  author={Chin-Yun Yu and Christopher Mitcheltree and Alistair Carson and Stefan Bilbao and Joshua D. Reiss and György Fazekas},
  booktitle={International Conference on Digital Audio Effects (DAFx)},
  year={2024}
}
```
