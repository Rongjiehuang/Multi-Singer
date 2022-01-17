# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging
import os

from multiprocessing import Manager

import numpy as np

from torch.utils.data import Dataset

from utils import find_files
from utils import read_hdf5



class AudioMelEmbedDataset(Dataset):  #读取audio与mel h5数据集
    """PyTorch compatible audio and mel dataset."""  # 读取音频、梅尔频谱数据集

    def __init__(self,
                 root_file,
                 feat_type='librosa',
                 audio_length_threshold=None,
                 frames_threshold=None,
                 use_f0=False,
                 use_chroma=False,
                 use_utt_id=False,
                 allow_cache=False,
                 eval=False
                 ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            audio_load_fn (func): Function to load audio file.
            mel_load_fn (func): Function to load feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        if eval:
            files = sorted(find_files(root_file, "*.h5"))
        else:
            files = []
            with open(root_file, encoding='utf-8') as f:
                for line in f:
                    files.append(line.strip().split('|')[1])
            files = sorted(files)

        audio_load_fn = lambda x: read_hdf5(x, "wav")  # 读取音频文件映射函数: h5["wav"]
        feat_load_fn = lambda x: read_hdf5(x, "mel")   # 读取梅尔文件映射函数: h5["mel"]
        embed_load_fn = lambda x: read_hdf5(x, "embed")  # 读取embed文件映射函数: h5["embed"]

        if feat_type == "world":  # 读取world提取特征
            feat_load_fn = lambda x: read_hdf5(x, "feats")  # 使用world提取特征 h5["feats"]

        # filter by threshold
        if audio_length_threshold is not None:  # 设置音频最长长度
            audio_lengths = [audio_load_fn(f).shape[0] for f in files]
            idxs = [idx for idx in range(len(files)) if audio_lengths[idx] > audio_length_threshold]   # 过滤得到音频长度超过阈值
            if len(files) != len(idxs):
                logging.warning(f"Some files are filtered by audio length threshold "
                                f"({len(files)} -> {len(idxs)}).")
            files = [files[idx] for idx in idxs]
        if frames_threshold is not None:
            frames = [feat_load_fn(f).shape[0] for f in files]
            idxs = [idx for idx in range(len(files)) if frames[idx] > frames_threshold]   # 过滤得到梅尔长度超过阈值
            if len(files) != len(idxs):
                logging.warning(f"Some files are filtered by mel length threshold "
                                f"({len(files)} -> {len(idxs)}).")
            files = [files[idx] for idx in idxs]

        # assert the number of files
        assert len(files) != 0, f"Not found any audio files in ${root_file}."

        self.files = files
        self.audio_load_fn = audio_load_fn
        self.feat_load_fn = feat_load_fn
        self.embed_load_fn = embed_load_fn

        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
        self.use_f0 = use_f0
        self.use_chroma = use_chroma
        self.use_utt_id = use_utt_id
        self.allow_cache = allow_cache

        if use_f0:
            self.f0_origin_load_fn = lambda x: read_hdf5(x, "f0_origin")
            # self.uv_load_fn = lambda x: read_hdf5(x, "uv")
            # self.f0_load_fn = lambda x: read_hdf5(x, "f0")

        if use_chroma:
            self.chroma_load_fn =lambda x: read_hdf5(x, "chroma")

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).
            ndarray: embed (256, ).
        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        audio = self.audio_load_fn(self.files[idx])
        feat = self.feat_load_fn(self.files[idx])
        embed = self.embed_load_fn(self.files[idx])
        items = {'audio':audio, 'feat':feat, 'embed':embed}

        if self.use_utt_id:
            items['utt_id'] = self.utt_ids[idx]
        if self.use_chroma:
            items['chroma'] = self.chroma_load_fn(self.files[idx])
        if self.use_f0:
            # items['f0'] = self.f0_load_fn(self.files[idx])
            items['f0_origin'] = self.f0_origin_load_fn(self.files[idx])
            # items['uv'] = self.uv_load_fn(self.files[idx])

        if self.allow_cache:
            self.caches[idx] = items

        return items  # 返回音频与梅尔频谱，以及其他可选参数(F0)

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.files)

class AudioDataset(Dataset):
    """PyTorch compatible audio dataset."""

    def __init__(self,
                 root_dir,
                 audio_query="*-wave.npy",
                 audio_length_threshold=None,
                 audio_load_fn=np.load,
                 return_utt_id=False,
                 allow_cache=False,
                 ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            audio_load_fn (func): Function to load audio file.
            audio_length_threshold (int): Threshold to remove short audio files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [idx for idx in range(len(audio_files)) if audio_lengths[idx] > audio_length_threshold]
            if len(audio_files) != len(idxs):
                logging.waning(f"some files are filtered by audio length threshold "
                               f"({len(audio_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.return_utt_id = return_utt_id
        if ".npy" in audio_query:
            self.utt_ids = [os.path.basename(f).replace("-wave.npy", "") for f in audio_files]
        else:
            self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in audio_files]
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio (T,).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])

        if self.return_utt_id:
            items = utt_id, audio
        else:
            items = audio

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class MelDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(self,
                 root_dir,
                 mel_query="*-feats.npy",
                 mel_length_threshold=None,
                 mel_load_fn=np.load,
                 return_utt_id=False,
                 allow_cache=False,
                 ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            mel_query (str): Query to find feature files in root_dir.
            mel_load_fn (func): Function to load feature file.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of the mel files
        mel_files = sorted(find_files(root_dir, mel_query))

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [idx for idx in range(len(mel_files)) if mel_lengths[idx] > mel_length_threshold]
            if len(mel_files) != len(idxs):
                logging.warning(f"Some files are filtered by mel length threshold "
                                f"({len(mel_files)} -> {len(idxs)}).")
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mel files in ${root_dir}."

        self.mel_files = mel_files
        self.mel_load_fn = mel_load_fn
        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        if ".npy" in mel_query:
            self.utt_ids = [os.path.basename(f).replace("-feats.npy", "") for f in mel_files]
        else:
            self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(mel_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        mel = self.mel_load_fn(self.mel_files[idx])

        if self.return_utt_id:
            items = utt_id, mel
        else:
            items = mel

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.mel_files)

