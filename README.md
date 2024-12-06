# 4(Feed-Forward)A-2A: an emulator plugin for LA-2A optical compressor

4A-2A is a VST plugin that emulates the LA-2A optical compressor using a feed-forward digital compressor.
The accompanying abstract, [Emulating LA-2A Optical Compressor With a Feed-Forward Digital Compressor Using the Newton-Raphson Method](media/DMRN_19__4A_2A.pdf), is accepted at DMRN+19.

We implement a mapping function $\mathbb{R} \to \mathbb{R}^5$ that maps the peak reduction of the LA-2A to the five parameters of the feed-forward compressor. The mapping function is learnt using the Newton-Raphson method on the [SignalTrain](https://doi.org/10.5281/zenodo.3824876) dataset.

## Interface

![](media/gui.png)

The plugin comes with six sliders, which are:

1. `Threshold` - The threshold of the compressor in dB.
2. `Ratio` - The ratio of the compressor (1 to 20).
3. `Attack` - The attack time of the compressor in ms (0.1 to 100).
4. `Release` - The release time of the compressor in ms (100 to 1000).
5. `Make-up` - The make-up gain of the compressor in dB.
6. `Peak Reduction` - The emulated peak reduction of the LA-2A in dB. Modifying this slider will change the other five sliders, to simulate the characteristics of the LA-2A.

## Installation

The pre-built VST3 binaries for Windows and macOS can be found in the [releases]().
One can also build the plugin from source using the [Projucer](https://juce.com/download/) project file.

## Training code
Coming soon.

## Citation
If you use this plugin in your research, please cite it as:

```
@inproceedings{yu2024emulating,
  title={Emulating LA-2A Optical Compressor With a Feed-Forward Digital Compressor Using the Newton-Raphson Method},
  author={Chin-Yun Yu and György Fazekas},
  booktitle={Proceedings of the Digital Music Research Network Workshop},
  year={2024}
}
```