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
# from stylegan2_pytorch.version import __version__
# from stylegan2_pytorch.diff_augment import DiffAugment

from vector_quantize_pytorch import VectorQuantize

from PIL import Image
from pathlib import Path

from .model_adgen import VggStyleEncoder, LinearBlock, MLP, Decoder, Conv2dBlock
from .vgg import VGG
import os
import torchvision.models.vgg as models



# n_downsample = 4
class StyleGan2Gen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, dim, style_dim, n_downsample, n_res, mlp_dim, activ='relu', pad_type='reflect'):
        super(StyleGan2Gen, self).__init__()

        style_downsample = 4
        style_dim = 2048
        SP_input_nc = 24#8
        input_dim = 3
        style_pool_feature = dim*(2**style_downsample)
        self.enc_style = VGGStyleEnc()
        self.enc_style1 = DownSampleEnc(style_downsample, input_dim*SP_input_nc, dim, 'in', activ, pad_type=pad_type)
        self.enc_style_pooling = PoolingModule(style_pool_feature, style_dim)

        # content encoder
        # n_downsample = 4
        input_dim = 3#18
        dim = 32
        # Foe dimension
        self.enc_content = DownSampleEnc(n_downsample, input_dim, dim, 'in', activ, pad_type=pad_type)

        num_layers = n_downsample + 1
        network_capacity = 16 # fpmax = 512 so the max filter size is clipped to 512
        # #Things to control: Above + noise vector

        self.gen = Generator(num_layers, style_dim, network_capacity = network_capacity)


    def forward(self, img_A, img_B, sem_B, noise):
        # noise = image_noise(batch_size, image_size, device=self.rank)
        # reconstruct an image
        # print("Input ", torch.min(img_A), torch.max(img_A))
        content = self.enc_content(img_A)
        # print("Content  ", torch.min(content), torch.max(content))
        # print("Content  ", content.shape)

        sem_B_shape = sem_B.shape
        sem_B = sem_B.contiguous().view(sem_B.shape[0], -1, sem_B.shape[-2], sem_B.shape[-1])
        style = self.enc_style1(sem_B)
        style = self.enc_style_pooling(style)
        # print("Noise ", noise.shape)        
        images_recon = self.gen(content, style.squeeze(), noise)
        # print("image_recon ", torch.min(images_recon), torch.max(images_recon))
        # print("images_recon ", images_recon.shape, content.shape)
        return images_recon

class StyleGan2Gen1(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, dim, style_dim, n_downsample, n_res, mlp_dim, activ='relu', pad_type='reflect'):
        super(StyleGan2Gen1, self).__init__()

        n_downsample = 4

        style_dim = 576
        SP_input_nc = 24#8
        input_dim = 3
        self.enc_style = VggStyleEncoder(3, input_dim, dim, int(style_dim/SP_input_nc), norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        # n_downsample = 4
        input_dim = 3#18
        dim = 32
        # Foe dimension
        self.enc_content = DownSampleEnc(n_downsample, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.fc = LinearBlock(style_dim, style_dim, norm='none', activation=activ)

        latent_dim = 256
        num_layers = n_downsample + 1
        network_capacity = 16*2 # fpmax = 512 so the max filter size is clipped to 512
        # #Things to control: Above + noise vector
        self.mlp = MLP(style_dim, num_layers*latent_dim, mlp_dim, 3, norm='none', activ=activ)

        self.gen = Generator(num_layers, latent_dim, network_capacity = network_capacity)
        self.latent_dim = latent_dim


    def forward(self, img_A, img_B, sem_B, noise):
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
        images_recon = self.gen(content, style, noise)

        # sem_B_shape = sem_B.shape
        # sem_B = sem_B.contiguous().view(sem_B.shape[0], -1, sem_B.shape[-2], sem_B.shape[-1])
        # style = self.enc_style1(sem_B)
        # style = self.enc_style_pooling(style)
        # # print("Noise ", noise.shape)        
        # images_recon = self.gen(content, style.squeeze(), noise)
        # # print("image_recon ", torch.min(images_recon), torch.max(images_recon))
        # # print("images_recon ", images_recon.shape, content.shape)
        return images_recon


class StyleGan2Gen2(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, dim, style_dim, n_downsample, n_res, mlp_dim, activ='relu', pad_type='reflect'):
        super(StyleGan2Gen2, self).__init__()

        n_downsample = 4

        style_dim = 192
        SP_input_nc = 24#8
        input_dim = 3
        self.enc_style = VggStyleEncoder(3, input_dim, dim, int(style_dim/SP_input_nc), norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        # n_downsample = 4
        input_dim = 3#18
        dim = 32
        # Foe dimension
        self.enc_content = DownSampleEnc(n_downsample, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.fc = LinearBlock(style_dim, style_dim, norm='none', activation=activ)

        latent_dim = 256
        num_layers = n_downsample + 1
        network_capacity = 16*2 # fpmax = 512 so the max filter size is clipped to 512
        # #Things to control: Above + noise vector
        self.mlp = MLP(style_dim, num_layers*latent_dim, mlp_dim, 3, norm='none', activ=activ)

        self.gen = Generator(num_layers, latent_dim, network_capacity = network_capacity)
        self.latent_dim = latent_dim


    def forward(self, img_A, img_B, sem_B, noise):
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
        images_recon = self.gen(content, style, noise)

        return images_recon


class StyleGan2Gen3(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, dim, style_dim, n_downsample, n_res, mlp_dim, activ='relu', pad_type='reflect'):
        super(StyleGan2Gen, self).__init__()

        style_downsample = 4
        style_dim = 2048
        SP_input_nc = 24#8
        input_dim = 3
        style_pool_feature = dim*(2**style_downsample)
        self.enc_style = VGGStyleEnc()
        self.enc_style1 = DownSampleEnc(style_downsample, input_dim*SP_input_nc, dim, 'in', activ, pad_type=pad_type)
        self.enc_style_pooling = PoolingModule(style_pool_feature, style_dim)

        # content encoder
        # n_downsample = 4
        input_dim = 3#18
        dim = 32
        # Foe dimension
        self.enc_content = DownSampleEnc(n_downsample, input_dim, dim, 'in', activ, pad_type=pad_type)

        num_layers = n_downsample + 1
        network_capacity = 16 # fpmax = 512 so the max filter size is clipped to 512
        # #Things to control: Above + noise vector

        self.gen = Generator(num_layers, style_dim, network_capacity = network_capacity)


    def forward(self, img_A, img_B, sem_B, noise):
        # noise = image_noise(batch_size, image_size, device=self.rank)
        # reconstruct an image
        # print("Input ", torch.min(img_A), torch.max(img_A))
        content = self.enc_content(img_A)
        # print("Content  ", torch.min(content), torch.max(content))
        # print("Content  ", content.shape)

        sem_B_shape = sem_B.shape
        sem_B = sem_B.contiguous().view(sem_B.shape[0], -1, sem_B.shape[-2], sem_B.shape[-1])
        style = self.enc_style1(sem_B)
        style = self.enc_style_pooling(style)
        # print("Noise ", noise.shape)        
        images_recon = self.gen(content, style.squeeze(), noise)
        # print("image_recon ", torch.min(images_recon), torch.max(images_recon))
        # print("images_recon ", images_recon.shape, content.shape)
        return images_recon



# stylegan2 classes
class VGGStyleEnc(nn.Module):
    def __init__(self):
        super(VGGStyleEnc, self).__init__()
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('/nitthilan/data/ADGAN/data/vgg19-dcbb9e9d.pth'))
        self.vgg = vgg19.features

    def forward(self, x):
        return x

class PoolingModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PoolingModule, self).__init__()
        self.model = []
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(input_dim, output_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class DownSampleEnc(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, norm, activ, pad_type):
        super(DownSampleEnc, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        # self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if exists(self.upsample):
            x = self.upsample(x)

        inoise = inoise[:x.shape[0], :x.shape[3], :x.shape[2], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        # print("Noise shape ", noise1.shape, inoise.shape, x.shape)

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb

class Generator(nn.Module):
    def __init__(self, num_layers, latent_dim, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        # self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers #int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]


        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        print("Filter configuration ", filters, in_out_pairs)

        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.blocks.append(block)

    def forward(self, x, styles, input_noise):
        # batch_size = style.shape[0]

        # if self.no_const:
        #     avg_style = styles.mean(dim=1)[:, :, None, None]
        #     x = self.to_initial_block(avg_style)
        # else:
        #     x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        # print("input styles ", styles.shape, x.shape)
        for style, block, attn in zip(styles, self.blocks, self.attns):
            if exists(attn):
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
        rgb = nn.functional.tanh(rgb)
        # print("rgb ", rgb.shape)
        return rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x
class StyleDiscriminator(nn.Module):
    def __init__(self, num_layers=4, network_capacity = 16, fq_layers = [], fq_dict_size = 256, attn_layers = [], transparent = False, fmap_max = 512):
        super().__init__()

        n_downsample = 4
        num_layers = n_downsample + 1
        network_capacity = 16*2 # fpmax = 512 so the max filter size is clipped to 512
 
        # num_layers = int(log2(image_size) - 1)
        num_init_filters = 6 if not transparent else 8

        blocks = []
        filters = [num_init_filters] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        # self.flatten = Flatten()
        # self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            if exists(q_block):
                x, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        # x = self.flatten(x)
        # x = self.to_logit(x)
        x = self.sigmoid(x)
        return x #.squeeze(), quantize_loss





def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2D(x, f, normalized=True)

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def default(value, d):
    return value if exists(value) else d

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    return el if isinstance(el, list) else [el]

def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()

def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x

def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x

def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x

def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}


