#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import logging
import os
from encoder import inference as encoder
import librosa
import numpy as np
import soundfile as sf
import yaml
import random
from tqdm import tqdm
from multiprocessing.pool import Pool

from datasets import AudioDataset
from frontend.audio_preprocess import logmelfilterbank, pitchfeats, f0_to_coarse
from frontend.audio_world_process import world_feature_extract, convert_continuos_f0, low_pass_filter
from utils import write_hdf5
from utils import simple_table



def normalize(S):
    return np.clip((S + 100) / 100, -2, 2)


def extract_feats(wav, outdir, utt_id, config):

    wav = wav / np.abs(wav).max() * 0.5
    h5_file = os.path.join(outdir, f"{utt_id}.h5")
    if config['feat_type'] == 'librosa':
        mel = logmelfilterbank(wav, config)  #
        frames = len(mel)
        mel = normalize(mel) * 2
        # mel = melspectrogram(x, config).T
        write_hdf5(h5_file, "mel", mel.astype(np.float32))
        if config["use_chroma"]:
            chromagram = librosa.feature.chroma_stft(wav,
                                                     sr=config["sampling_rate"],
                                                     hop_length=config["hop_size"])
            write_hdf5(h5_file, "chroma", chromagram.T.astype(np.float32))

        if config["use_f0"]:
            f0 = pitchfeats(wav, config)
            write_hdf5(h5_file, "f0_origin", f0.astype(np.float))

        if config["use_embed"]:
            wav_torch = torch.from_numpy(wav)
            preprocessed_wav = encoder.preprocess_wav_torch(wav_torch)
            embed = encoder.embed_utterance_torch_preprocess(preprocessed_wav)
            embed = embed.detach().numpy()
            write_hdf5(h5_file, "embed", embed.astype(np.float32))

    elif config['feat_type'] == 'world':
        feats = world_feature_extract(wav, config)
        frames = len(feats)
        write_hdf5(h5_file, "feats", feats.astype(np.float32))

    else:
        raise NotImplementedError("Currently, only 'world'、'librosa' are supported.")

    audio = np.pad(wav, (0, config["fft_size"]), mode="edge")
    audio = audio[:frames * config["hop_size"]]
    assert frames * config["hop_size"] == len(audio)

    write_hdf5(h5_file, "wav", audio.astype(np.float32))

    return utt_id, h5_file, frames, len(audio)


def write2file(values, config, outdir):
    test_nums = config['test_num']
    train_text = open(os.path.join(outdir, 'train.txt'), 'w', encoding='utf-8')
    dev_text = open(os.path.join(outdir, 'dev.txt'), 'w', encoding='utf-8')

    for v in values[:test_nums]:
        dev_text.write('|'.join([str(x) for x in v]) + '\n')
    for v in values[test_nums:]:
        train_text.write('|'.join([str(x) for x in v]) + '\n')

    mel_frames = sum([int(m[2]) for m in values])
    timesteps = sum([int(m[3]) for m in values])
    sr = config['sampling_rate']
    hours = timesteps / sr / 3600
    logging.info('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
        len(values), mel_frames, timesteps, hours))
    logging.info('Max mel frames length: {}'.format(max(int(m[2]) for m in values)))
    logging.info('Max audio timesteps length: {}'.format(max(m[3] for m in values)))


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in parallel_wavegan/bin/preprocess.py).")
    parser.add_argument("--inputdir",'-i', type=str, required=True,
                        help="directory including wav files. you need to specify either scp or inputdir.")
    parser.add_argument("--dumpdir",'-o', type=str,required=True,
                        help="directory to dump feature files.")
    parser.add_argument("--config",'-c', type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning('Skip DEBUG/INFO messages')

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    # check arguments
    if args.inputdir is None:
        raise ValueError("Please specify either --rootdir or --wav-scp.")

    # get dataset
    assert args.inputdir is not None
    dataset = AudioDataset(
            args.inputdir, "*.wav",
            audio_load_fn=sf.read,
            return_utt_id=True,
    )
    if config["use_embed"]:
        print("Preparing the encoder...")
        encoder.load_model(config["enc_model_fpath"],preprocess=True)

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # process each data
    futures = []
    p = Pool(int(os.getenv('N_PROC', os.cpu_count())))

    simple_table([
        ('Data Path', args.inputdir),
        ('Preprocess Path', args.dumpdir),
        ('Config File', args.config),
        ('CPU Usage', os.cpu_count())
    ])



    for utt_id, (audio, fs) in tqdm(dataset):
        # check
        assert len(audio.shape) == 1, \
            f"{utt_id} seems to be multi-channel signal."
        assert np.abs(audio).max() <= 1.0, \
            f"{utt_id} seems to be different from 16 bit PCM."
        assert fs == config["sampling_rate"], \
            f"{utt_id} seems to have a different sampling rate."

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(audio,
                                            top_db=config["trim_threshold_in_db"],
                                            frame_length=config["trim_frame_size"],
                                            hop_length=config["trim_hop_size"])

        if "sampling_rate_for_feats" not in config:
            x = audio
            sampling_rate = config["sampling_rate"]
            hop_size = config["hop_size"]
        else:

            x = librosa.resample(audio, fs, config["sampling_rate_for_feats"])
            sampling_rate = config["sampling_rate_for_feats"]
            assert config["hop_size"] * config["sampling_rate_for_feats"] % fs == 0, \
                "hop_size must be int value. please check sampling_rate_for_feats is correct."
            hop_size = config["hop_size"] * config["sampling_rate_for_feats"] // fs

        config["sampling_rate"] = sampling_rate
        config["hop_size"] = hop_size

        feats_dir = os.path.join(args.dumpdir, 'feats')
        os.makedirs(feats_dir, exist_ok=True)

        futures.append(p.apply_async(extract_feats, args=(x, feats_dir, utt_id, config)))

    p.close()
    values = []
    for future in tqdm(futures):
        values.append(future.get())

    random.seed(2020)
    random.shuffle(values)

    write2file(values, config, args.dumpdir)

if __name__ == "__main__":
    main()
