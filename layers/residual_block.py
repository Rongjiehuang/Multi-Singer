# -*- coding: utf-8 -*-

"""Residual block module in WaveNet.

This code is modified from https://github.com/r9y9/wavenet_vocoder.

"""

import math

import torch
import torch.nn.functional as F


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(in_channels, out_channels,
                                        kernel_size=1, padding=0,
                                        dilation=1, bias=bias)


class ResidualBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(self,
                 kernel_size=3,
                 residual_channels=64,
                 gate_channels=128,
                 skip_channels=64,
                 aux_channels=80, # 条件输入维度:80维Mel频谱
                 dropout=0.0,
                 dilation=1,
                 bias=True,
                 use_causal_conv=False
                 ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            dropout (float): Dropout probability.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            use_causal_conv (bool): Whether to use use_causal_conv or non-use_causal_conv convolution.

        """
        super(ResidualBlock, self).__init__()
        self.dropout = dropout
        # no future time stamps available
        if use_causal_conv:
            padding = (kernel_size - 1) * dilation
        else:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv

        # dilation conv
        self.conv = Conv1d(residual_channels, gate_channels, kernel_size,
                           padding=padding, dilation=dilation, bias=bias)

        # local conditioning  加入条件输入 (B, aux_channels, T) -> (B, gate_channels, T)
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None

        # conv output is split into two groups   GAU门输出拆为两部分: residual与skip connection
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)  # x经过膨胀卷积  (B, residual_channels, T) -> (B, gate_channels, T)

        # remove future time steps if use_causal_conv conv 去除x中residual未来的时间步
        x = x[:, :, :residual.size(-1)] if self.use_causal_conv else x

        # split into two part for gated activation  (B, gate_channels, T) -> 2*(B, gate_channels/2, T)
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)  # 拆分后分别通过tanh与sigmoid

        # local conditioning: WaveNet中的条件输入,同样经过拆分后附加到xa,xb上
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)  # condition经过一层卷积
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # for skip connection
        s = self.conv1x1_skip(x)

        # for residual connection  卷积输出+残差快
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

        return x, s  # 返回residual 和 skip



class ResidualEmbeddingBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(self,
                 kernel_size=3,
                 residual_channels=64,
                 gate_channels=128,
                 skip_channels=64,
                 aux_channels=80, # 条件输入维度:80维Mel频谱
                 embed_channels=256,
                 dropout=0.0,
                 dilation=1,
                 bias=True,
                 use_causal_conv=False
                 ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            dropout (float): Dropout probability.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            use_causal_conv (bool): Whether to use use_causal_conv or non-use_causal_conv convolution.

        """
        super(ResidualEmbeddingBlock, self).__init__()
        self.dropout = dropout
        # no future time stamps available
        if use_causal_conv:
            padding = (kernel_size - 1) * dilation
        else:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv

        # dilation conv
        self.conv = Conv1d(residual_channels, gate_channels, kernel_size,
                           padding=padding, dilation=dilation, bias=bias)

        # local conditioning  加入条件输入 (B, aux_channels, T) -> (B, gate_channels, T)
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None

        if aux_channels > 0:
            self.conv1x1_embed = Conv1d1x1(embed_channels, gate_channels, bias=False)
        else:
            self.conv1x1_embed = None

        # conv output is split into two groups   GAU门输出拆为两部分: residual与skip connection
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, x, c, embed):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).
            embed (Tensor): Local conditioning auxiliary tensor (B, embed_channels, T).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)  # x经过膨胀卷积  (B, residual_channels, T) -> (B, gate_channels, T)

        # remove future time steps if use_causal_conv conv 去除x中residual未来的时间步
        x = x[:, :, :residual.size(-1)] if self.use_causal_conv else x

        # split into two part for gated activation  (B, gate_channels, T) -> 2*(B, gate_channels/2, T)
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)  # 拆分后分别通过tanh与sigmoid

        # local conditioning: WaveNet中的条件输入,同样经过拆分后附加到xa,xb上
        if c is not None and embed is not None:
            assert self.conv1x1_aux is not None
            assert self.conv1x1_embed is not None
            c = self.conv1x1_aux(c)  # condition经过一层卷积
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)

            embed = self.conv1x1_embed(embed)  # condition经过一层卷积
            ea, eb = embed.split(embed.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca + ea, xb + cb + eb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # for skip connection
        s = self.conv1x1_skip(x)

        # for residual connection  卷积输出+残差快
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

        return x, s  # 返回residual 和 skip
