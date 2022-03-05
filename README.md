# Multi-Singer: Fast Multi-Singer Singing Voice Vocoder With A Large-Scale Corpus

PyTorch Implementation of (ACM MM'21)[Multi-Singer: Fast Multi-Singer Singing Voice Vocoder With A Large-Scale Corpus](https://dl.acm.org/doi/pdf/10.1145/3474085.3475437).



## Requirements
See requirements in requirement.txt:
- linux
- python 3.6 
- pytorch 1.0+
- librosa
- json, tqdm, logging



## Getting started

#### Apply recipe to your own dataset

- Put any wav files in data directory
- Edit configuration in config/config.yaml


## 1. Pretrain
[Use our checkpoint](https://github.com/Rongjiehuang/Multi-Singer/blob/main/pretrained1.pt), or\
you can also train the encoder on your own [here](https://github.com/dipjyoti92/speaker_embeddings_GE2E), and set the ```enc_model_fpath``` in config/config.yaml. Please set params as those in ```encoder/params_data``` and ```encoder/params_model```.

## 2. Preprocess

Extract mel-spectrogram

```python
python preprocess.py -i data/wavs -o data/feature -c config/config.yaml
```

`-i`  your audio folder

`-o` output acoustic feature folder

`-c` config file



## 3. Train

Training conditioned on mel-spectrogram

```python
python train.py -i data/feature -o checkpoints/ --config config/config.yaml
```

`-i` acoustic feature folder

`-o` directory to save checkpoints

`-c`  config file

## 4. Inference

```python
python inference.py -i data/feature -o outputs/  -c checkpoints/*.pkl -g config/config.yaml
```

`-i` acoustic feature folder

`-o` directory to save generated speech

`-c` checkpoints file

`-c`  config file

## 5. Singing Voice Synthesis
For Singing Voice Synthesis:
- Take [modified FastSpeech 2](https://github.com/ming024/FastSpeech2) for mel-spectrogram synthesis
- Use synthesized mel-spectrogram in Multi-Singer for waveform synthesis.

## Checkpoint
[Trained on OpenSinger](https://github.com/Rongjiehuang/Multi-Singer/blob/main/Basic.pkl)


## Acknowledgements
[GE2E](https://github.com/dipjyoti92/speaker_embeddings_GE2E)\
[FastSpeech 2](https://github.com/ming024/FastSpeech2)\
[Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)


## Citation
```
@inproceedings{huang2021multi,
  title={Multi-Singer: Fast Multi-Singer Singing Voice Vocoder With A Large-Scale Corpus},
  author={Huang, Rongjie and Chen, Feiyang and Ren, Yi and Liu, Jinglin and Cui, Chenye and Zhao, Zhou},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={3945--3954},
  year={2021}
}
```

## Question
Feel free to contact me at rongjiehuang@zju.edu.cn
