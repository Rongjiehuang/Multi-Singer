
import logging
import os

from multiprocessing import Manager

import numpy as np

from torch.utils.data import Dataset

from utils import find_files
from utils import read_hdf5
import torch



class Feats_Collater(object):
    """Customized collater for Pytorch DataLoader in training."""  # 收集函数collator

    def __init__(self,
                 batch_max_steps=20480,
                 out_dim=1,
                 hop_size=256,
                 aux_context_window=2,
                 use_noise_input=False,
                 use_f0=False,
                 use_chroma=False
                 ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.

        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.out_dim = out_dim
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input
        self.use_f0 = use_f0
        self.use_chroma = use_chroma

        # set useful values in random cutting  随机截取长度
        self.start_offset = aux_context_window  # 开始偏移位置 = 窗大小
        self.end_offset = -(self.batch_max_frames + aux_context_window)  # 结束偏移位置 = -(最大帧长 + 窗大小)
        self.mel_threshold = self.batch_max_frames + 2 * aux_context_window

    def __call__(self, batch):
        # check length
        # batch = [self._adjust_length(*b) for b in batch if len(b[1]) > self.mel_threshold]
        xs, cs = [b['audio'] for b in batch], [b['feat'] for b in batch]  # batch 包含audio & feat(mel)

        # make batch with random cut  随机裁剪窗
        c_lengths = [len(c) for c in cs]
        start_frames = np.array([np.random.randint(
            self.start_offset, cl + self.end_offset) for cl in c_lengths])
        x_starts = start_frames * self.hop_size                                   # audio 起始
        x_ends = x_starts + self.batch_max_steps                                  # audio 结束
        c_starts = start_frames - self.aux_context_window                         # mel 起始
        c_ends = start_frames + self.batch_max_frames + self.aux_context_window   # mel 结束
        y_batch = [x[start: end] for x, start, end in zip(xs, x_starts, x_ends)]  # 得到audio
        c_batch = [c[start: end] for c, start, end in zip(cs, c_starts, c_ends)]  # 得到mel

        # convert each batch to tensor, asuume that each item in batch has the same length—————将numpy转为tensor
        y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)
        c_batch = torch.tensor(c_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')

        batchs = {'audios': y_batch, 'feats': c_batch}  ###################    得到 batch["audio"] 与 batch["feats"]   ###################

        if self.use_f0:
            # f0s = [b['f0'] for b in batch if 'f0' in b]
            # f0_batch = [f0[start: end] for f0, start, end in zip(f0s, c_starts, c_ends)]
            # f0_batch = torch.tensor(f0_batch, dtype=torch.long)
            # batchs['f0s'] = f0_batch

            f0_origins = [b['f0_origin'] for b in batch if "f0_origin" in b]
            f0_origins_batch = [f0[start+self.aux_context_window: end-self.aux_context_window] for f0, start, end in zip(f0_origins, c_starts, c_ends)]
            f0_origins_batch = torch.tensor(f0_origins_batch, dtype=torch.float)
            batchs['f0_origins'] = f0_origins_batch

            # vus = [b['uv']  for b in batch if "uv" in b]
            # vus_batch = [vu[start: end] for vu, start, end in zip(vus, c_starts, c_ends)]
            # vus_batch = torch.tensor(vus_batch, dtype=torch.long)
            # batchs['uvs'] = vus_batch

        if self.use_chroma:
            chromas = [b['chroma'] for b in batch if 'chroma' in b]
            chroma_batch = [chromas[start: end] for chroma, start, end in zip(chromas, c_starts, c_ends)]
            chroma_batch = torch.tensor(chroma_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')
            batchs['chromas'] = chroma_batch
        # make input noise signal batch tensor
        if self.use_noise_input:
            # z_batch = torch.randn(y_batch.size())  # (B, 1, T)
            z_batch = torch.randn(y_batch.size(0), self.out_dim, y_batch.size(2) // self.out_dim)  # (B, 1, T)
            batchs['noise'] = z_batch

        return batchs


class Embeds_Collater(object):
    """Customized collater for Pytorch DataLoader in training."""  # 收集函数collator

    def __init__(self,
                 batch_max_steps=20480,
                 out_dim=1,
                 hop_size=256,
                 aux_context_window=2,
                 use_noise_input=False,
                 use_f0=False,
                 use_chroma=False
                 ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.

        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.out_dim = out_dim
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input
        self.use_f0 = use_f0
        self.use_chroma = use_chroma

        # set useful values in random cutting  随机截取长度
        self.start_offset = aux_context_window  # 开始偏移位置 = 窗大小
        self.end_offset = -(self.batch_max_frames + aux_context_window)  # 结束偏移位置 = -(最大帧长 + 窗大小)
        self.mel_threshold = self.batch_max_frames + 2 * aux_context_window

    def __call__(self, batch):
        # check length
        # batch = [self._adjust_length(*b) for b in batch if len(b[1]) > self.mel_threshold]
        xs, cs = [b['audio'] for b in batch], [b['feat'] for b in batch]  # batch 包含audio & feat(mel)
        embed = [b['embed'] for b in batch]

        # make batch with random cut  随机裁剪窗
        c_lengths = [len(c) for c in cs]
        start_frames = np.array([np.random.randint(
            self.start_offset, cl + self.end_offset) for cl in c_lengths])
        x_starts = start_frames * self.hop_size                                   # audio 起始
        x_ends = x_starts + self.batch_max_steps                                  # audio 结束
        c_starts = start_frames - self.aux_context_window                         # mel 起始
        c_ends = start_frames + self.batch_max_frames + self.aux_context_window   # mel 结束
        y_batch = [x[start: end] for x, start, end in zip(xs, x_starts, x_ends)]  # 得到audio
        c_batch = [c[start: end] for c, start, end in zip(cs, c_starts, c_ends)]  # 得到mel

        # convert each batch to tensor, asuume that each item in batch has the same length—————将numpy转为tensor
        y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)
        c_batch = torch.tensor(c_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')
        embed_batch = torch.tensor(embed, dtype=torch.float).unsqueeze(-1)  # (B, 128) -> (B, 128, 1)

        batchs = {'audios': y_batch, 'feats': c_batch, 'embed': embed_batch}  ###################    得到 batch["audio"] 与 batch["feats"]   ###################

        if self.use_f0:
            # f0s = [b['f0'] for b in batch if 'f0' in b]
            # f0_batch = [f0[start: end] for f0, start, end in zip(f0s, c_starts, c_ends)]
            # f0_batch = torch.tensor(f0_batch, dtype=torch.long)
            # batchs['f0s'] = f0_batch

            f0_origins = [b['f0_origin'] for b in batch if "f0_origin" in b]
            f0_origins_batch = [f0[start+self.aux_context_window: end-self.aux_context_window] for f0, start, end in zip(f0_origins, c_starts, c_ends)]
            f0_origins_batch = torch.tensor(f0_origins_batch, dtype=torch.float)
            batchs['f0_origins'] = f0_origins_batch

            # vus = [b['uv']  for b in batch if "uv" in b]
            # vus_batch = [vu[start: end] for vu, start, end in zip(vus, c_starts, c_ends)]
            # vus_batch = torch.tensor(vus_batch, dtype=torch.long)
            # batchs['uvs'] = vus_batch

        if self.use_chroma:
            chromas = [b['chroma'] for b in batch if 'chroma' in b]
            chroma_batch = [chromas[start: end] for chroma, start, end in zip(chromas, c_starts, c_ends)]
            chroma_batch = torch.tensor(chroma_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')
            batchs['chromas'] = chroma_batch
        # make input noise signal batch tensor
        if self.use_noise_input:
            # z_batch = torch.randn(y_batch.size())  # (B, 1, T)
            z_batch = torch.randn(y_batch.size(0), 1, y_batch.size(2) // self.out_dim)  # (B, 1, T)
            batchs['noise'] = z_batch

        return batchs