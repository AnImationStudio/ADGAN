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

import torch
from torch import nn, einsum
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange, repeat
from kornia.filters import filter2D

import torchvision
from torchvision import transforms
from stylegan2_pytorch.version import __version__
from stylegan2_pytorch.diff_augment import DiffAugment

from vector_quantize_pytorch import VectorQuantize

from PIL import Image
from pathlib import Path

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



