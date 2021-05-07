import os
import sys
import math
import fire
import json

from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np
import math
from typing import List
import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan


from .model_adgen import VggStyleEncoder, LinearBlock, MLP, Decoder, Conv2dBlock


class StyleGan2Gen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, dim, style_dim, n_downsample, n_res, mlp_dim, activ='relu', pad_type='reflect'):
        super(StyleGan2Gen, self).__init__()

        n_downsample = 2
        style_dim = 576

        # style encoder
        input_dim = 3
        SP_input_nc = 24#8
        self.enc_style = VggStyleEncoder(3, input_dim, dim, int(style_dim/SP_input_nc), norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        input_dim = 3#18
        self.enc_content = DownSampleEnc(n_downsample, input_dim, dim, 'in', activ, pad_type=pad_type)
        # input_dim = 3
        # self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        self.fc = LinearBlock(style_dim, style_dim, norm='none', activation=activ)

        num_layers = n_downsample + 1
        latent_dim = 256
        network_capacity = 16*2
        #Things to control: Above + noise vector

        # fusion module
        self.mlp = MLP(style_dim, num_layers*latent_dim, mlp_dim, 3, norm='none', activ=activ)

        self.gen = Generator(num_layers, latent_dim, network_capacity = network_capacity)
        self.latent_dim = latent_dim

        # self.noise = torch.FloatTensor(1, 44, 64, 1).uniform_(0., 1.)
        self.register_buffer('noise', torch.FloatTensor(256, 44, 64, 1).uniform_(0., 1.))


    def forward(self, img_A, img_B, sem_B):
        # noise = image_noise(batch_size, image_size, device=self.rank)
        # reconstruct an image
        # print("Input ", torch.min(img_A), torch.max(img_A))
        content = self.enc_content(img_A)
        # print("Content  ", torch.min(content), torch.max(content))
        # print("Content  ", content.shape)

        style = self.enc_style(img_B, sem_B)
        # print("Style1 ", torch.min(style), torch.max(style))
        style = self.fc(style.view(style.size(0), -1))
        # print("Style2 ", torch.min(style), torch.max(style))
        style = torch.unsqueeze(style, 2)
        style = torch.unsqueeze(style, 3)
        # print("Style1 ", style.shape)
        style = self.mlp(style)
        style = style.view(style.size(0), -1, self.latent_dim)

        # images_recon = self.decode(content, style)
        images_recon = self.gen(content, style, self.noise)
        # print("image_recon ", torch.min(images_recon), torch.max(images_recon))
        # print("images_recon ", images_recon.shape, content.shape)
        return images_recon


def siren_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', c: float = 6):
    r"""Fills the input `Tensor` with values according to the method
    described in ` Implicit Neural Representations with Periodic Activation
    Functions.` - Sitzmann, Martel et al. (2020), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \sqrt{\frac{6}{\text{fan\_mode}}}
    Also known as Siren initialization.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> siren.init.siren_uniform_(w, mode='fan_in', c=6)
    :param tensor: an n-dimensional `torch.Tensor`
    :type tensor: torch.Tensor
    :param mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing
        ``'fan_in'`` preserves the magnitude of the variance of the weights in
        the forward pass. Choosing ``'fan_out'`` preserves the magnitudes in
        the backwards pass.s
    :type mode: str, optional
    :param c: value used to compute the bound. defaults to 6
    :type c: float, optional
    """
    fan = _calculate_correct_fan(tensor, mode)
    std = 1 / math.sqrt(fan)
    bound = math.sqrt(c) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.
        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                'input to forward() must be torch.xTensor')


class SIREN(nn.Module):
    def __init__(self, layers: List[int], in_features: int,
                 out_features: int,
                 w0: float = 1.0,
                 w0_initial: float = 30.0,
                 bias: bool = True,
                 initializer: str = 'siren',
                 c: float = 6):
        """
        SIREN model from the paper [Implicit Neural Representations with
        Periodic Activation Functions](https://arxiv.org/abs/2006.09661).
        :param layers: list of number of neurons in each hidden layer
        :type layers: List[int]
        :param in_features: number of input features
        :type in_features: int
        :param out_features: number of final output features
        :type out_features: int
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        :param w0_initial: `w0` of first layer. defaults to 30 (as used in the
            paper)
        :type w0_initial: float, optional
        :param bias: whether to use bias or not. defaults to
            True
        :type bias: bool, optional
        :param initializer: specifies which initializer to use. defaults to
            'siren'
        :type initializer: str, optional
        :param c: value used to compute the bound in the siren intializer.
            defaults to 6
        :type c: float, optional
        # References:
            -   [Implicit Neural Representations with Periodic Activation
                 Functions](https://arxiv.org/abs/2006.09661)
        """
        super(SIREN, self).__init__()
        self._check_params(layers)
        self.layers = [nn.Linear(in_features, layers[0], bias=bias), Sine(
            w0=w0_initial)]

        for index in range(len(layers) - 1):
            self.layers.extend([
                nn.Linear(layers[index], layers[index + 1], bias=bias),
                Sine(w0=w0)
            ])

        self.layers.append(nn.Linear(layers[-1], out_features, bias=bias))
        self.network = nn.Sequential(*self.layers)

        if initializer is not None and initializer == 'siren':
            for m in self.network.modules():
                if isinstance(m, nn.Linear):
                    siren_uniform_(m.weight, mode='fan_in', c=c)

    @staticmethod
    def _check_params(layers):
        assert isinstance(layers, list), 'layers should be a list of ints'
        assert len(layers) >= 1, 'layers should not be empty'

    def forward(self, X):
        return self.network(X)

