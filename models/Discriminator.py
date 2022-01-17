# -*- coding: utf-8 -*-



"""Multi-Singer Modules."""

import numpy as np
import torch
import logging
from layers import Conv1d



class Unconditional_Discriminator(torch.nn.Module):
    """Unconditional Discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=10,
                 conv_channels=64,
                 dilation_factor=1,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 bias=True,
                 use_weight_norm=True,
                 ):
        """Initialize Unconditional Discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """

        super(Unconditional_Discriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert dilation_factor > 0, "Dilation factor must be > 0."
        self.conv_layers = torch.nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):  # (B, 1, T) -> (B, 64, T)
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor ** i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                Conv1d(conv_in_channels, conv_channels,
                       kernel_size=kernel_size, padding=padding,
                       dilation=dilation, bias=bias),
                getattr(torch.nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params)
            ]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        last_conv_layer = Conv1d(   # (B, 64, T) -> (B, 1, T)
            conv_in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_layers += [last_conv_layer]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        for f in self.conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class SingerConditional_Discriminator(torch.nn.Module):
    """SingerConditional Discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=256,
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=1024,
                 bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 model_hidden_size=256,
                 model_num_layers=3
                 ):
        """Initialize SingerConditional Discriminator module.
        """
        super(SingerConditional_Discriminator, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=model_hidden_size,
                            hidden_size=model_hidden_size,
                            num_layers=model_num_layers)

        self.linear = torch.nn.Linear(model_hidden_size,1)
        self.relu = torch.nn.ReLU()

        # add first layer   (B, 1, T) ->  (B, channels, T)
        self.layers = torch.nn.ModuleList()
        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels, np.prod(kernel_sizes), bias=bias),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers    (B, channels, T) -> (B, channels*downsample_scale[0], T/downsample_scale[0])
        # -> ...  -> (B, channels*downsample_scale[0]*...*downsample_scale[3], T/(downsample_scale[0]*...*downsample_scale[3]))
        # -> ...  -> (B, channels*downsample_scale[0]*...*downsample_scale[3], T/product(downsample_scale))
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs, out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 5,
                        groups=in_chs // 4,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
            in_chs = out_chs

        # add final layers  (B, channels*downsample_scale[0]*...*downsample_scale[3], T/product(downsample_scale)) -> (B, channels*downsample_scale[0]*...*downsample_scale[3], T/product(downsample_scale))
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs, out_chs, kernel_sizes[0],
                    padding=(kernel_sizes[0] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [  # (B, channels*downsample_scale[0]*...*downsample_scale[3], T/product(downsample_scale)) -> (B, 1,  T/product(downsample_scale))
            torch.nn.Conv1d(
                out_chs, out_channels, kernel_sizes[1],
                padding=(kernel_sizes[1] - 1) // 2,
                bias=bias,
            ),
        ]


    def forward(self, x, embed):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            embed (Tensor): Local conditioning auxiliary features (B, C ,1).

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """

        for f in self.layers:  # (B, 1, T) -> (B, 256, T/prob(downscale))
            x = f(x)

        frames_batch = x.permute(2,0,1) # (B, 256, T/prob(downscale))  -> (seq_len, B, 256)
        output, (hn, cn) = self.lstm(frames_batch)  # output: (seq_len, batch, model_embedding_size)  hidden: (layers, batch, model_embedding_size)

        p = output[-1] + embed.squeeze(2) # (batch, model_embedding_size) + (batch, model_embedding_size)
        p = self.relu(self.linear(p))  # (batch, 1)
        return p

