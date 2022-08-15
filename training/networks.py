# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import functools
import torch
import torch.nn as nn
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

from util_classes import * 
import torch.nn.functional as F
from util_functions import random_affine_matrix, get_random_crops

import dnnlib
import math
from collections import OrderedDict

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# feature encoder
@persistence.persistent_class
class FeatureEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=7):
        super().__init__()        
        
        encoder = []
        encoder += [Conv2dLayer(input_nc, ngf, kernel_size=1)]
        mult_ins = [1, 2, 4, 4, 8, 8, 8]
        mult_outs = [2, 4, 4, 8, 8, 8, 8]
        for i in range(n_downsampling):
            mult_in = mult_ins[i]
            mult_out = mult_outs[i]
            encoder += [Conv2dLayer(ngf * mult_in, ngf * mult_out, kernel_size=3, down=2)]
        self.model = nn.Sequential(*encoder)
            
    def forward(self, x):
        x = self.model(x)      

        return x    


@persistence.persistent_class
class ResBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.conv0 = Conv2dLayer(in_channels, out_channels, kernel_size=3, activation=activation, up=up, down=down, bias=bias, 
                                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=channels_last)
        self.conv1 = Conv2dLayer(out_channels, out_channels, kernel_size=3, activation=activation, bias=bias, resample_filter=resample_filter,
                                conv_clamp=conv_clamp, channels_last=channels_last)
        self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=up, down=down, resample_filter=resample_filter,
                                conv_clamp=conv_clamp, channels_last=channels_last)

    def forward(self, x):
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        x = y.add_(x)
        return x   

@persistence.persistent_class
class ResBlock_partialconv(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.down = down

        self.conv0 = Conv2dLayer_partialconv(in_channels, out_channels, kernel_size=3, activation=activation, up=up, down=down, bias=bias, 
                                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=channels_last)
        self.conv1 = Conv2dLayer_partialconv(out_channels, out_channels, kernel_size=3, activation=activation, bias=bias, resample_filter=resample_filter,
                                conv_clamp=conv_clamp, channels_last=channels_last)
        self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=up, down=down, resample_filter=resample_filter,
                                conv_clamp=conv_clamp, channels_last=channels_last)

    def forward(self, x, mask):
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x, mask)
        if self.down == 2:
            mask = torch.nn.functional.interpolate(mask,scale_factor=0.5)
            mask = (mask==1).to(x.dtype).to(x.device)
        x = self.conv1(x, mask, gain=np.sqrt(0.5))
        x = y.add_(x)
        return x   


@persistence.persistent_class
class ConstEncoderNetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4):
        super().__init__()        
        
        encoder = []
        encoder += [Conv2dLayer(input_nc, ngf, kernel_size=1)]
        mult_ins = [1, 2, 4, 4, 4, 8]
        mult_outs = [2, 4, 4, 4, 8, 8]
        for i in range(n_downsampling):
            mult_in = mult_ins[i]
            mult_out = mult_outs[i]
            encoder += [Conv2dLayer(ngf * mult_in, ngf * mult_out, kernel_size=3, down=2)]

        self.model = nn.Sequential(*encoder)
            
    def forward(self, x):
        x = self.model(x)      

        return x    


#----------------------------------------------------------------------------

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, self.block_size, stride=self.block_size)
        return unfolded_x.view(n, c * self.block_size ** 2, h // self.block_size, w // self.block_size)


class Dense(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(in_channels, out_channels)


    def forward(self, x):
        x = x.permute((0, 2, 3, 1))
        # x = x.view(b*h*w, -1)
        out = self.linear(x)
        out = out.permute((0, 3, 1, 2))
        out = self.bn(out)
        out = self.activation(out)
        return out

class Attention(nn.Module):
    def __init__(self, ch, use_sn):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.theta = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        if use_sn:
            self.theta = spectral_norm(self.theta)
            self.phi = spectral_norm(self.phi)
            self.g = spectral_norm(self.g)
            self.o = spectral_norm(self.o)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

#----------------------------------------------------------------------------


########################### patch discriminator architecture ####################################
class RandomSpatialTransformer:
    def __init__(self, opt, bs):
        self.opt = opt
        #self.resample_transformation(bs)


    def create_affine_transformation(self, ref, rot, sx, sy, tx, ty):
        return torch.stack([-ref * sx * torch.cos(rot), -sy * torch.sin(rot), tx,
                            -ref * sx * torch.sin(rot), sy * torch.cos(rot), ty], axis=1)

    def resample_transformation(self, bs, device, reflection=None, rotation=None, scale=None, translation=None):
        dev = device
        zero = torch.zeros((bs), device=dev)
        if reflection is None:
            #if "ref" in self.opt.random_transformation_mode:
            ref = torch.round(torch.rand((bs), device=dev)) * 2 - 1
            #else:
            #    ref = 1.0
        else:
            ref = reflection

        if rotation is None:
            #if "rot" in self.opt.random_transformation_mode:
            max_rotation = 30 * math.pi / 180
            rot = torch.rand((bs), device=dev) * (2 * max_rotation) - max_rotation
            #else:
            #    rot = 0.0
        else:
            rot = rotation

        if scale is None:
            #if "scale" in self.opt.random_transformation_mode:
            min_scale = 1.0
            max_scale = 1.0
            sx = torch.rand((bs), device=dev) * (max_scale - min_scale) + min_scale
            sy = torch.rand((bs), device=dev) * (max_scale - min_scale) + min_scale
            #else:
            #    sx, sy = 1.0, 1.0
        else:
            sx, sy = scale

        tx, ty = zero, zero

        A = torch.stack([ref * sx * torch.cos(rot), -sy * torch.sin(rot), tx,
                         ref * sx * torch.sin(rot), sy * torch.cos(rot), ty], axis=1)
        return A.view(bs, 2, 3)

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.dim() == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


from torch.autograd import Function
from torch.utils.cpp_extension import load

def is_custom_kernel_supported():
    version_str = str(torch.version.cuda).split(".")
    major = version_str[0]
    minor = version_str[1]
    return int(major) >= 10 and int(minor) >= 1

if is_custom_kernel_supported():
    print("Loading custom kernel...")
    module_path = os.path.dirname(__file__)
    upfirdn2d_op = load(
        'upfirdn2d',
        sources=[
            os.path.join(module_path, 'upfirdn2d.cpp'),
            os.path.join(module_path, 'upfirdn2d_kernel.cu'),
        ],
        verbose=True
    )

use_custom_kernel = is_custom_kernel_supported()

@persistence.persistent_class
class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None

@persistence.persistent_class
class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn2d_v2(input, kernel, up=1, down=1, pad=(0, 0)):
    global use_custom_kernel
    if use_custom_kernel:
        out = UpFirDn2d.apply(
            input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
        )
    else:
        out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    bs, ch, in_h, in_w = input.shape
    minor = 1
    kernel_h, kernel_w = kernel.shape

    #assert kernel_h == 1 and kernel_w == 1

    #print("original shape ", input.shape, up_x, down_x, pad_x0, pad_x1)

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    if up_x > 1 or up_y > 1:
        out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])

    #print("after padding ", out.shape)
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    #print("after reshaping ", out.shape)

    if pad_x0 > 0 or pad_x1 > 0 or pad_y0 > 0 or pad_y1 > 0:
        out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])

    #print("after second padding ", out.shape)
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    #print("after trimming ", out.shape)

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )

    #print("after reshaping", out.shape)
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)

    #print("after conv ", out.shape)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )

    out = out.permute(0, 2, 3, 1)

    #print("after permuting ", out.shape)

    out = out[:, ::down_y, ::down_x, :]

    out = out.view(bs, ch, out.size(1), out.size(2))

    #print("final shape ", out.shape)

    return out

use_custom_kernel = is_custom_kernel_supported()


@persistence.persistent_class
class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, reflection_pad=False):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.reflection = reflection_pad
        if self.reflection:
            self.reflection_pad = nn.ReflectionPad2d((pad[0], pad[1], pad[0], pad[1]))
            self.pad = (0, 0)

    def forward(self, input):
        if self.reflection:
            input = self.reflection_pad(input)
        out = upfirdn2d_v2(input, self.kernel, pad=self.pad)

        return out

@persistence.persistent_class
class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    global use_custom_kernel
    if use_custom_kernel:
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
    else:
        dims = [1, -1] + [1] * (input.dim() - 2)
        bias = bias.view(*dims)
        return F.leaky_relu(input + bias, negative_slope) * scale

@persistence.persistent_class
class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

@persistence.persistent_class
class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, lr_mul=1.0,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2) * lr_mul

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

@persistence.persistent_class
class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        pad=None,
        reflection_pad=False,
    ):
        layers = []

        if downsample:
            factor = 2
            if pad is None:
                pad = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (pad + 1) // 2
            pad1 = pad // 2

            layers.append(("Blur", Blur(blur_kernel, pad=(pad0, pad1), reflection_pad=reflection_pad)))

            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2 if pad is None else pad
            if reflection_pad:
                layers.append(("RefPad", nn.ReflectionPad2d(self.padding)))
                self.padding = 0


        layers.append(("Conv",
                       EqualConv2d(
                           in_channel,
                           out_channel,
                           kernel_size,
                           padding=self.padding,
                           stride=stride,
                           bias=bias and not activate,
                       ))
        )

        if activate:
            if bias:
                layers.append(("Act", FusedLeakyReLU(out_channel)))

            else:
                layers.append(("Act", ScaledLeakyReLU(0.2)))

        super().__init__(OrderedDict(layers))

    def forward(self, x):
        out = super().forward(x)
        return out

@persistence.persistent_class
class ResBlock_PD(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], reflection_pad=False, pad=None, downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, reflection_pad=reflection_pad, pad=pad)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample, blur_kernel=blur_kernel, reflection_pad=reflection_pad, pad=pad)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, blur_kernel=blur_kernel, activate=False, bias=False
        )

    def forward(self, input):
        #print("before first resnet layeer, ", input.shape)
        out = self.conv1(input)
        #print("after first resnet layer, ", out.shape)
        out = self.conv2(out)
        #print("after second resnet layer, ", out.shape)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

@persistence.persistent_class
class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            if input.dim() > 2:
                out = F.conv2d(input, self.weight[:, :, None, None] * self.scale)
            else:
                out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            if input.dim() > 2:
                out = F.conv2d(input, self.weight[:, :, None, None] * self.scale,
                               bias=self.bias * self.lr_mul
                )
            else:
                out = F.linear(
                    input, self.weight * self.scale, bias=self.bias * self.lr_mul
                )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

@persistence.persistent_class
class BaseNetwork(torch.nn.Module):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     return parser

    def __init__(self, 
                netPatchD_scale_capacity=4.0,
                netPatchD_max_nc=256+128,
                patch_size=64,
                max_num_tiles=8,
                use_antialias=True):
        super().__init__()
        self.opt = dnnlib.EasyDict(
            netPatchD_scale_capacity=netPatchD_scale_capacity,
            netPatchD_max_nc=netPatchD_max_nc,
            patch_size=patch_size,
            max_num_tiles=max_num_tiles,
            use_antialias=use_antialias,        
        )

    def print_architecture(self, verbose=False):
        name = type(self).__name__
        result = '-------------------%s---------------------\n' % name
        total_num_params = 0
        for i, (name, child) in enumerate(self.named_children()):
            num_params = sum([p.numel() for p in child.parameters()])
            total_num_params += num_params
            if verbose:
                result += "%s: %3.3fM\n" % (name, (num_params / 1e6))
            for i, (name, grandchild) in enumerate(child.named_children()):
                num_params = sum([p.numel() for p in grandchild.parameters()])
                if verbose:
                    result += "\t%s: %3.3fM\n" % (name, (num_params / 1e6))
        result += '[Network %s] Total number of parameters : %.3f M\n' % (name, total_num_params / 1e6)
        result += '-----------------------------------------------\n'
        print(result)

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def collect_parameters(self, name):
        params = []
        for m in self.modules():
            if type(m).__name__ == name:
                params += list(m.parameters())
        return params

    def fix_and_gather_noise_parameters(self):
        params = []
        device = next(self.parameters()).device
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                assert m.image_size is not None, "One forward call should be made to determine size of noise parameters"
                m.fixed_noise = torch.nn.Parameter(torch.randn(m.image_size[0], 1, m.image_size[2], m.image_size[3], device=device))
                params.append(m.fixed_noise)
        return params

    def remove_noise_parameters(self, name):
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                m.fixed_noise = None

    def forward(self, x):
        return x

@persistence.persistent_class
class BasePatchDiscriminator(BaseNetwork):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument("--netPatchD_scale_capacity", default=4.0, type=float)
    #     parser.add_argument("--netPatchD_max_nc", default=256 + 128, type=int)
    #     parser.add_argument("--patch_size", default=128, type=int)
    #     parser.add_argument("--max_num_tiles", default=8, type=int)
    #     parser.add_argument("--patch_random_transformation",
    #                         type=util.str2bool, nargs='?', const=True, default=False)
    #     return parser

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.visdom = util.Visualizer(opt)

    def needs_regularization(self):
        return False

    def extract_features(self, patches):
        raise NotImplementedError()

    def discriminate_features(self, feature1, feature2):
        raise NotImplementedError()

    def apply_random_transformation(self, patches):
        B, ntiles, C, H, W = patches.size()
        patches = patches.view(B * ntiles, C, H, W)
        before = patches
        transformer = RandomSpatialTransformer(self.opt, B * ntiles)
        patches = transformer.forward_transform(patches, (self.opt.patch_size, self.opt.patch_size))
        #self.visdom.display_current_results({'before': before,
        #                                     'after': patches}, 0, save_result=False)
        return patches.view(B, ntiles, C, H, W)

    def sample_patches(self, img, indices):
        B, C, H, W = img.size()
        s = self.opt.patch_size
        if H % s > 0 or W % s > 0:
            y_offset = torch.randint(H % s, (), device=img.device)
            x_offset = torch.randint(W % s, (), device=img.device)
            img = img[:, :,
                      y_offset:y_offset + s * (H // s),
                      x_offset:x_offset + s * (W // s)]
        img = img.view(B, C, H//s, s, W//s, s)
        ntiles = (H // s) * (W // s)
        tiles = img.permute(0, 2, 4, 1, 3, 5).reshape(B, ntiles, C, s, s)
        if indices is None:
            indices = torch.randperm(ntiles, device=img.device)[:self.opt.max_num_tiles]
            return self.apply_random_transformation(tiles[:, indices]), indices
        else:
            return self.apply_random_transformation(tiles[:, indices])

    def forward(self, real, fake, fake_only=False):
        assert real is not None
        real_patches, patch_ids = self.sample_patches(real, None)
        if fake is None:
            real_patches.requires_grad_()
        real_feat = self.extract_features(real_patches)

        bs = real.size(0)
        if fake is None or not fake_only:
            pred_real = self.discriminate_features(
                real_feat,
                torch.roll(real_feat, 1, 1))
            pred_real = pred_real.view(bs, -1)


        if fake is not None:
            fake_patches = self.sample_patches(fake, patch_ids)
            #self.visualizer.display_current_results({'real_A': real_patches[0],
            #                                         'real_B': torch.roll(fake_patches, 1, 1)[0]}, 0, False, max_num_images=16)
            fake_feat = self.extract_features(fake_patches)
            pred_fake = self.discriminate_features(
                real_feat,
                torch.roll(fake_feat, 1, 1))
            pred_fake = pred_fake.view(bs, -1)

        if fake is None:
            return pred_real, real_patches
        elif fake_only:
            return pred_fake
        else:
            return pred_real, pred_fake

@persistence.persistent_class
class StyleGAN2PatchDiscriminator(BasePatchDiscriminator):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     BasePatchDiscriminator.modify_commandline_options(parser, is_train)
    #     return parser

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        channel_multiplier = self.opt.netPatchD_scale_capacity
        size = self.opt.patch_size
        channels = {
            4: min(self.opt.netPatchD_max_nc, int(256 * channel_multiplier)),
            8: min(self.opt.netPatchD_max_nc, int(128 * channel_multiplier)),
            16: min(self.opt.netPatchD_max_nc, int(64 * channel_multiplier)),
            32: int(32 * channel_multiplier),
            64: int(16 * channel_multiplier),
            128: int(8 * channel_multiplier),
            256: int(4 * channel_multiplier),
        }

        log_size = int(math.ceil(math.log(size, 2)))

        in_channel = channels[2 ** log_size]

        blur_kernel = [1, 3, 3, 1] if self.opt.use_antialias else [1]

        convs = [('0', ConvLayer(3, in_channel, 3))]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            layer_name = str(7 - i) if i <= 6 else "%dx%d" % (2 ** i, 2 ** i)
            convs.append((layer_name, ResBlock_PD(in_channel, out_channel, blur_kernel)))

            in_channel = out_channel

        convs.append(('5', ResBlock_PD(in_channel, self.opt.netPatchD_max_nc * 2, downsample=False)))
        convs.append(('6', ConvLayer(self.opt.netPatchD_max_nc * 2, self.opt.netPatchD_max_nc, 3, pad=0)))

        self.convs = nn.Sequential(OrderedDict(convs))

        out_dim = 1

        pairlinear1 = EqualLinear(channels[4] * 2 * 2 * 2, 2048, activation='fused_lrelu')
        pairlinear2 = EqualLinear(2048, 2048, activation='fused_lrelu')
        pairlinear3 = EqualLinear(2048, 1024, activation='fused_lrelu')
        pairlinear4 = EqualLinear(1024, out_dim)
        self.pairlinear = nn.Sequential(pairlinear1, pairlinear2, pairlinear3, pairlinear4)

    def extract_features(self, patches, aggregate=False):
        if patches.ndim == 5:
            B, T, C, H, W = patches.size()
            flattened_patches = patches.flatten(0, 1)
        else:
            B, C, H, W = patches.size()
            T = patches.size(1)
            flattened_patches = patches
        features = self.convs(flattened_patches)
        features = features.view(B, T, features.size(1), features.size(2), features.size(3))
        if aggregate:
            features = features.mean(1, keepdim=True).expand(-1, T, -1, -1, -1)
        return features.flatten(0, 1)

    def extract_layerwise_features(self, image):
        feats = [image]
        for m in self.convs:
            feats.append(m(feats[-1]))

        return feats

    def discriminate_features(self, feature1, feature2):
        feature1 = feature1.flatten(1)
        feature2 = feature2.flatten(1)
        out = self.pairlinear(torch.cat([feature1, feature2], dim=1))
        return out

    def forward(self, real, target, is_for_Gmain=False):
        if is_for_Gmain:
            real_feat = self.extract_features(real,aggregate=True).detach()
        else:
            real_feat = self.extract_features(real,aggregate=True)
        target_feat = self.extract_features(target)
        pred = self.discriminate_features(real_feat,target_feat)
        return pred


# only user patch as the inputs of the discriminator, do not use the average real feature @@@@
@persistence.persistent_class
class StyleGAN2PatchDiscriminator_V2(BasePatchDiscriminator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        channel_multiplier = self.opt.netPatchD_scale_capacity
        size = self.opt.patch_size
        channels = {
            4: min(self.opt.netPatchD_max_nc, int(256 * channel_multiplier)),
            8: min(self.opt.netPatchD_max_nc, int(128 * channel_multiplier)),
            16: min(self.opt.netPatchD_max_nc, int(64 * channel_multiplier)),
            32: int(32 * channel_multiplier),
            64: int(16 * channel_multiplier),
            128: int(8 * channel_multiplier),
            256: int(4 * channel_multiplier),
        }

        log_size = int(math.ceil(math.log(size, 2)))

        in_channel = channels[2 ** log_size]

        blur_kernel = [1, 3, 3, 1] if self.opt.use_antialias else [1]

        convs = [('0', ConvLayer(3, in_channel, 3))]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            layer_name = str(7 - i) if i <= 6 else "%dx%d" % (2 ** i, 2 ** i)
            convs.append((layer_name, ResBlock_PD(in_channel, out_channel, blur_kernel)))

            in_channel = out_channel

        convs.append(('5', ResBlock_PD(in_channel, self.opt.netPatchD_max_nc * 2, downsample=False)))
        convs.append(('6', ConvLayer(self.opt.netPatchD_max_nc * 2, self.opt.netPatchD_max_nc, 3, pad=0)))

        self.convs = nn.Sequential(OrderedDict(convs))

        out_dim = 1

        # pairlinear1 = EqualLinear(channels[4] * 2 * 2 * 2, 2048, activation='fused_lrelu')
        pairlinear1 = EqualLinear(channels[4] * 2 * 2, 2048, activation='fused_lrelu')
        pairlinear2 = EqualLinear(2048, 2048, activation='fused_lrelu')
        pairlinear3 = EqualLinear(2048, 1024, activation='fused_lrelu')
        pairlinear4 = EqualLinear(1024, out_dim)
        self.pairlinear = nn.Sequential(pairlinear1, pairlinear2, pairlinear3, pairlinear4)

    def extract_features(self, patches, aggregate=False):
        if patches.ndim == 5:
            B, T, C, H, W = patches.size()
            flattened_patches = patches.flatten(0, 1)
        else:
            B, C, H, W = patches.size()
            T = patches.size(1)
            flattened_patches = patches
        features = self.convs(flattened_patches)
        features = features.view(B, T, features.size(1), features.size(2), features.size(3))
        if aggregate:
            features = features.mean(1, keepdim=True).expand(-1, T, -1, -1, -1)
        return features.flatten(0, 1)

    def extract_layerwise_features(self, image):
        feats = [image]
        for m in self.convs:
            feats.append(m(feats[-1]))

        return feats

    # def discriminate_features(self, feature1, feature2):
    #     feature1 = feature1.flatten(1)
    #     feature2 = feature2.flatten(1)
    #     out = self.pairlinear(torch.cat([feature1, feature2], dim=1))
    #     return out

    def discriminate_features(self, feature1):
        feature1 = feature1.flatten(1)
        # feature2 = feature2.flatten(1)
        # out = self.pairlinear(torch.cat([feature1, feature2], dim=1))
        out = self.pairlinear(feature1)
        return out

    # def forward(self, real, target, is_for_Gmain=False):
    #     if is_for_Gmain:
    #         real_feat = self.extract_features(real,aggregate=True).detach()
    #     else:
    #         real_feat = self.extract_features(real,aggregate=True)
    #     target_feat = self.extract_features(target)
    #     pred = self.discriminate_features(real_feat,target_feat)
    #     return pred
    def forward(self, target):

        target_feat = self.extract_features(target)
        pred = self.discriminate_features(target_feat)
        return pred

############################### use spade in the stylegan decoder #########################

@misc.profiled_function
def spade_modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    spade_styles    = None,
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]
    if spade_styles is not None:
        misc.assert_shape(spade_styles, [batch_size, in_channels,x.shape[2],x.shape[3]])

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    # if demodulate and fused_modconv:
    #     w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    # if not fused_modconv:
    if spade_styles is not None:
        x = x * (spade_styles.to(x.dtype)+styles.to(x.dtype).reshape(batch_size,-1,1,1)) / 2
    else:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
    x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
    if demodulate and noise is not None:
        x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
    elif demodulate:
        x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
    elif noise is not None:
        x = x.add_(noise.to(x.dtype))
    return x

    # Execute as one fused op using grouped convolution.
    # with misc.suppress_tracer_warnings(): # this value will be treated as a constant
    #     batch_size = int(batch_size)
    # misc.assert_shape(x, [batch_size, in_channels, None, None])
    # x = x.reshape(1, -1, *x.shape[2:])
    # w = w.reshape(-1, in_channels, kh, kw)
    # x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    # x = x.reshape(batch_size, -1, *x.shape[2:])
    # if noise is not None:
    #     x = x.add_(noise)
    # return x

@persistence.persistent_class
class Spade_Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'relu',       # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1, no_act=False):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None

        if not no_act:
            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        return x


@persistence.persistent_class
class Spade_Conv2dLayer_partialconv(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'relu',       # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

        mask_weight = torch.ones([1,1,kernel_size, kernel_size]).to(memory_format=memory_format)
        self.register_buffer('mask_weight', mask_weight)


    def forward(self, x, mask, gain=1, no_act=False):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None

        if not no_act:
            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        mask_w = self.mask_weight
        mask_inverse = (mask==0)
        x_mask = conv2d_resample.conv2d_resample(x=mask, w=mask_w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)
        x_mask = x_mask.masked_fill_(mask_inverse,1.0)
        x = x / x_mask

        return x


@persistence.persistent_class
class Spade_Norm_Block(torch.nn.Module):
    def __init__(self,
        in_channels,
        norm_channels,
    ):
        super().__init__()
        self.conv_mlp = Spade_Conv2dLayer(in_channels, norm_channels, kernel_size=3, bias=False)
        self.conv_mlp_act = nn.ReLU()
        self.conv_gamma = Spade_Conv2dLayer(norm_channels, norm_channels, kernel_size=3, bias=False)
        self.conv_beta = Spade_Conv2dLayer(norm_channels, norm_channels, kernel_size=3, bias=False)

        self.param_free_norm = nn.InstanceNorm2d(norm_channels, affine=False)

    def forward(self, x, denorm_feats):
        normalized = self.param_free_norm(x)
        actv = self.conv_mlp(denorm_feats, no_act=True)
        actv = self.conv_mlp_act(actv)
        gamma = self.conv_gamma(actv, no_act=True)
        beta = self.conv_beta(actv, no_act=True)

        out = normalized * (1+gamma) + beta
        return out


@persistence.persistent_class
class StyleEncoderNetworkV18(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4):
        super().__init__()
        encoder = []
        encoder += [Conv2dLayer(input_nc, ngf, kernel_size=1)]
        mult_ins = [1, 2, 4]
        mult_outs = [2, 4, 8]
        for i in range(3):
            mult_in = mult_ins[i]
            mult_out = mult_outs[i]
            encoder += [Dense(ngf * mult_in, ngf * mult_in),
                        Conv2dLayer(ngf * mult_in, ngf * mult_out, kernel_size=3, down=2)]
        mult_ins = [8, 8, 8]
        mult_outs = [8, 8, 8]
        for i in range(3):
            mult_in = mult_ins[i]
            mult_out = mult_outs[i]
            encoder += [Dense(ngf * mult_in, ngf * mult_in), 
                        Conv2dLayer(ngf * mult_in, ngf * mult_out, kernel_size=3)]

        encoder += [nn.AdaptiveAvgPool2d(1)]
        self.model = nn.Sequential(*encoder)
        self.fc = FullyConnectedLayer(output_nc, output_nc)

        feat_enc = []
        # feat_enc += [Conv2dLayer(3, ngf, kernel_size=3)]
        feat_enc += [Conv2dLayer(6, ngf, kernel_size=3)]
        mult_ins = [1, 1, 1]
        mult_outs = [1, 1, 1]
        for i in range(3):
            mult_in = mult_ins[i]
            mult_out = mult_outs[i]
            feat_enc += [Conv2dLayer(ngf * mult_in, ngf * mult_out, kernel_size=3, down=2)]

        self.feat_enc = nn.Sequential(*feat_enc)

    def forward(self, x, const_input):
        const_feats = []
        for _, module in enumerate(self.feat_enc):
            const_input = module(const_input)
            const_feats.append(const_input)

        for _, module in enumerate(self.model):
            x = module(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, const_feats


@persistence.persistent_class
class ToRGBLayerV18(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False, is_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.is_last = is_last

        if self.is_last:
            self.m_weight1 = torch.nn.Parameter(torch.randn([1, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
            self.m_bias1 = torch.nn.Parameter(torch.zeros([1]))

            self.m_weight2 = torch.nn.Parameter(torch.randn([1, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
            self.m_bias2 = torch.nn.Parameter(torch.zeros([1]))


    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain

        upper_mask = None
        lower_mask = None
        if self.is_last:
            upper_mask = modulated_conv2d(x=x, weight=self.m_weight1, styles=styles, demodulate=False, fused_modconv=fused_modconv)
            upper_mask = bias_act.bias_act(upper_mask, self.m_bias1.to(x.dtype), clamp=self.conv_clamp, act='sigmoid')

            lower_mask = modulated_conv2d(x=x, weight=self.m_weight2, styles=styles, demodulate=False, fused_modconv=fused_modconv)
            lower_mask = bias_act.bias_act(lower_mask, self.m_bias2.to(x.dtype), clamp=self.conv_clamp, act='sigmoid')

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x, upper_mask, lower_mask

###########################  512 ###############################
@persistence.persistent_class
class ToRGBLayerV18_512(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False, is_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.is_last = is_last

        if self.is_last:
            self.m_weight1 = torch.nn.Parameter(torch.randn([in_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
            self.m_bias1 = torch.nn.Parameter(torch.zeros([in_channels]))
            self.m_weight1_1 = torch.nn.Parameter(torch.randn([1, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
            self.m_bias1_1 = torch.nn.Parameter(torch.zeros([1]))

            self.m_weight2 = torch.nn.Parameter(torch.randn([in_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
            self.m_bias2 = torch.nn.Parameter(torch.zeros([in_channels]))
            self.m_weight2_1 = torch.nn.Parameter(torch.randn([1, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
            self.m_bias2_1 = torch.nn.Parameter(torch.zeros([1]))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain

        upper_mask = None
        lower_mask = None
        if self.is_last:
            upper_mask = modulated_conv2d(x=x, weight=self.m_weight1, styles=styles, fused_modconv=fused_modconv)
            upper_mask = bias_act.bias_act(upper_mask, self.m_bias1.to(x.dtype), clamp=self.conv_clamp)
            upper_mask = modulated_conv2d(x=upper_mask, weight=self.m_weight1_1, styles=styles, demodulate=False, fused_modconv=fused_modconv)
            upper_mask = bias_act.bias_act(upper_mask, self.m_bias1_1.to(x.dtype), clamp=self.conv_clamp, act='sigmoid')

            lower_mask = modulated_conv2d(x=x, weight=self.m_weight2, styles=styles, fused_modconv=fused_modconv)
            lower_mask = bias_act.bias_act(lower_mask, self.m_bias2.to(x.dtype), clamp=self.conv_clamp)
            lower_mask = modulated_conv2d(x=lower_mask, weight=self.m_weight2_1, styles=styles, demodulate=False, fused_modconv=fused_modconv)
            lower_mask = bias_act.bias_act(lower_mask, self.m_bias2_1.to(x.dtype), clamp=self.conv_clamp, act='sigmoid')

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x, upper_mask, lower_mask


@persistence.persistent_class
class Spade_ResBlockV4_512(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        spade_channels, 
        kernel_size     = 3,            # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
        resolution      = 256,
    ):
        super().__init__()
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.conv = Spade_Conv2dLayer(in_channels, in_channels, kernel_size=3, bias=False,
                                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=channels_last)
        self.conv0 = Spade_Conv2dLayer(in_channels, out_channels, kernel_size=3, bias=False, 
                                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=channels_last)
        self.conv1 = Spade_Conv2dLayer(out_channels, out_channels, kernel_size=3, bias=False, 
                                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=channels_last)
        self.skip = Spade_Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, 
                                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=channels_last)

        # if resolution == 128:
        #     feat_channels = 128*2
        # else:
        #     feat_channels = 64*2
        feat_channels = spade_channels
        self.spade_skip = Spade_Norm_Block(feat_channels, in_channels)
        self.spade0 = Spade_Norm_Block(feat_channels, in_channels)
        self.spade1 = Spade_Norm_Block(feat_channels, out_channels)

    def forward(self, x, denorm_feat):
        x = self.conv(x, no_act=True)

        y = self.skip(self.spade_skip(x,denorm_feat), gain=np.sqrt(0.5))
        x = self.conv0(self.spade0(x,denorm_feat))
        x = self.conv1(self.spade1(x,denorm_feat),gain=np.sqrt(0.5))
        
        x = y.add_(x)
        return x   

################################ ###############################


@persistence.persistent_class
class ToRGBLayerFull_v1_v4(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False, is_last=False, is_style=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.is_last = is_last
        self.is_style = is_style

        if self.is_last and self.is_style:
            self.m_weight1 = torch.nn.Parameter(torch.randn([6, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
            self.m_bias1 = torch.nn.Parameter(torch.zeros([6]))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain

        pred_parsing = None
        if self.is_last and self.is_style:
            pred_parsing = modulated_conv2d(x=x, weight=self.m_weight1, styles=styles, demodulate=False, fused_modconv=fused_modconv)
            pred_parsing = bias_act.bias_act(pred_parsing, self.m_bias1.to(x.dtype), clamp=self.conv_clamp)

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x, pred_parsing



@persistence.persistent_class
class ToRGBLayerFull_v1_v5(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False, is_last=False, is_style=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.is_last = is_last
        self.is_style = is_style

        if self.is_last and self.is_style:
            self.m_weight1 = torch.nn.Parameter(torch.randn([7, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
            self.m_bias1 = torch.nn.Parameter(torch.zeros([7]))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain

        pred_parsing = None
        if self.is_last and self.is_style:
            pred_parsing = modulated_conv2d(x=x, weight=self.m_weight1, styles=styles, demodulate=False, fused_modconv=fused_modconv)
            pred_parsing = bias_act.bias_act(pred_parsing, self.m_bias1.to(x.dtype), clamp=self.conv_clamp)

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x, pred_parsing


@persistence.persistent_class
class SynthesisBlockFull_v1_v4(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        is_style            = False,        # Is this the block in the sytle synthesis branch
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        # CONST here
        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayerFull_v1_v4(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last, 
                is_last=is_last, is_style=is_style)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

        # if self.resolution > 16:
        #     self.merge_conv = Conv2dLayer(out_channels + 64, out_channels, kernel_size=1,
        #                               resample_filter=resample_filter, channels_last=self.channels_last)
        if self.resolution > 32:
            self.merge_conv = Conv2dLayer(out_channels + 64, out_channels, kernel_size=1,
                                      resample_filter=resample_filter, channels_last=self.channels_last)

        self.spade_b512 = Spade_ResBlockV4_512(out_channels, out_channels, spade_channels=1)

    def forward(self, x, img, ws, pose_feature, cat_feat, parsing, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            # CONST here
            # x = self.const.to(dtype=dtype, memory_format=memory_format)
            # x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
            x = pose_feature.to(dtype=dtype, memory_format=memory_format)
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

            # add warped feature here
            if x.shape[2] > 32:
                x = torch.cat([x, cat_feat[str(x.shape[2])].to(dtype=dtype, memory_format=memory_format)], dim=1)
                x = self.merge_conv(x)
            
            x = self.spade_b512(x, parsing)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y, pred_parsing = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        # assert x.dtype == dtype
        # assert img is None or img.dtype == torch.float32
        return x, img, pred_parsing


@persistence.persistent_class
class SynthesisBlockFull_v1_v6(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        is_style            = False,        # Is this the block in the sytle synthesis branch
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        # CONST here
        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayerFull_v1_v5(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last, 
                is_last=is_last, is_style=is_style)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

        # if self.resolution > 16:
        #     self.merge_conv = Conv2dLayer(out_channels + 64, out_channels, kernel_size=1,
        #                               resample_filter=resample_filter, channels_last=self.channels_last)
        if self.resolution > 32:
            self.merge_conv = Conv2dLayer(out_channels + 64, out_channels, kernel_size=1,
                                      resample_filter=resample_filter, channels_last=self.channels_last)


    def forward(self, x, img, ws, pose_feature, cat_feat, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            # CONST here
            # x = self.const.to(dtype=dtype, memory_format=memory_format)
            # x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
            x = pose_feature.to(dtype=dtype, memory_format=memory_format)
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

            # add warped feature here
            if x.shape[2] > 32:
                x = torch.cat([x, cat_feat[str(x.shape[2])].to(dtype=dtype, memory_format=memory_format)], dim=1)
                x = self.merge_conv(x)
            
        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y, pred_parsing = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        # assert x.dtype == dtype
        # assert img is None or img.dtype == torch.float32
        return x, img, pred_parsing


@persistence.persistent_class
class SynthesisNetworkFull_v18(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 8 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(3, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        # fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 8 else 0
            out_channels = channels_dict[res]
            # use_fp16 = (res >= fp16_resolution)
            use_fp16 = False
            is_last = (res == self.img_resolution)
            block = SynthesisBlockFull_v1_v6(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, is_style=True, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

        res = self.block_resolutions[-2]
        in_channels = channels_dict[res]
        out_channels = channels_dict[res]

        self.spade_b256_1 = Spade_ResBlockV4_512(in_channels, out_channels, spade_channels=128)
        self.spade_b256_2 = Spade_ResBlockV4_512(in_channels, out_channels, spade_channels=128)

        res = self.block_resolutions[-1]
        in_channels = channels_dict[res//2]
        out_channels = channels_dict[res]
        self.texture_b512 = SynthesisBlockFull_v1_v4(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=True, is_style=False, use_fp16=False, **block_kwargs)

        spade_encoder = []
        ngf = 64
        spade_encoder += [Conv2dLayer(3, ngf, kernel_size=7, activation='relu')]
        spade_encoder += [ResBlock(ngf, ngf, kernel_size=4, activation='relu')]                 # 512
        spade_encoder += [ResBlock(ngf, ngf*2, kernel_size=4, activation='relu', down=2)]       # 256
        self.spade_encoder = nn.Sequential(*spade_encoder)


    def get_spade_feat(self, mask_512, denorm_mask, denorm_input):
        mask_512 = (mask_512>0.9).to(mask_512.device).to(mask_512.dtype)
        mask_256 = torch.nn.functional.interpolate(mask_512,scale_factor=0.5)
        denorm_mask_256 = torch.nn.functional.interpolate(denorm_mask,scale_factor=0.5)
        mask_256 = (mask_256>0.9).to(mask_512.device).to(mask_512.dtype)
        denorm_mask_256 = (denorm_mask_256>0.9).to(mask_512.device).to(mask_512.dtype)

        valid_mask = ((mask_256+denorm_mask_256)==2.0).to(mask_512.device).to(mask_512.dtype)
        res_mask = (mask_256-valid_mask).to(mask_512.device).to(mask_512.dtype)

        denorm_input = denorm_input * mask_512 - (1-mask_512)
        spade_denorm_feat = self.spade_encoder(denorm_input)
        spade_denorm_valid_feat = spade_denorm_feat * valid_mask

        valid_feat_sum = torch.sum(spade_denorm_valid_feat, dim=(2,3), keepdim=True)
        valid_mask_sum = torch.sum(valid_mask, dim=(2,3), keepdim=True)

        valid_index = (valid_mask_sum>10).to(mask_512.device).to(mask_512.dtype)
        valid_mask_sum = valid_mask_sum * valid_index + (256*256) * (1-valid_index)
        spade_average_feat = valid_feat_sum / valid_mask_sum

        spade_feat = spade_denorm_feat * (1-res_mask) + spade_average_feat * res_mask

        return spade_feat


    def forward(self, ws, pose_feat, cat_feat, denorm_upper_input, denorm_lower_input, denorm_upper_mask, \
                denorm_lower_mask, gt_parsing, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img, pred_parsing = block(x, img, cur_ws, pose_feat, cat_feat, force_fp32=True, **block_kwargs)
            if res == 256:
                x_256, img_256 = x.clone(), img.clone()

        if gt_parsing is not None:
            parsing_index = gt_parsing
        else:
            softmax = torch.nn.Softmax(dim=1)
            parsing_index = torch.argmax(softmax(pred_parsing.detach()), dim=1)[:,None,...].float()

        upper_mask = (parsing_index==1).float() + (parsing_index==4).float()
        lower_mask = (parsing_index==2).float() + (parsing_index==3).float()

        spade_upper_feat = self.get_spade_feat(upper_mask.detach(), denorm_upper_mask, denorm_upper_input)
        spade_lower_feat = self.get_spade_feat(lower_mask.detach(), denorm_lower_mask, denorm_lower_input)

        upper_mask_256 = torch.nn.functional.interpolate(upper_mask,scale_factor=0.5)
        lower_mask_256 = torch.nn.functional.interpolate(lower_mask,scale_factor=0.5)
        upper_mask_256 = (upper_mask_256>0.9).to(upper_mask.dtype).to(upper_mask.device)
        lower_mask_256 = (lower_mask_256>0.9).to(upper_mask.dtype).to(upper_mask.device)

        spade_feat = spade_upper_feat * upper_mask_256 + spade_lower_feat * lower_mask_256

        x_spade_256 = self.spade_b256_1(x_256, spade_feat)
        x_spade_256 = self.spade_b256_2(x_spade_256, spade_feat)
        # x_spade_256 = self.spade_b256_3(x_spade_256, spade_feat)
        # x_spade_256 = self.spade_b256_2(x_spade_256, parsing_index_256)
        # x_spade_256 = self.spade_b256_3(x_spade_256, spade_feat)

        cur_ws = block_ws[-1]
        _, finetune_img, _ = self.texture_b512(x_spade_256, img_256, cur_ws, pose_feat, cat_feat, parsing_index, \
                                                        force_fp32=True, **block_kwargs)

        return img, finetune_img, pred_parsing

@persistence.persistent_class
class GeneratorFull_v20(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetworkFull_v18(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

        self.const_encoding = ConstEncoderNetwork(input_nc=3 + 2, output_nc=512, ngf=64, n_downsampling=6) 
        self.style_encoding = StyleEncoderNetworkV18(input_nc=(10*3+5*3), output_nc=512, ngf=64, n_downsampling=6)
   
    def forward(self, z, c, retain, pose, denorm_upper_input, denorm_lower_input, denorm_upper_mask, denorm_lower_mask, \
         gt_parsing=None, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        pose_feat = self.const_encoding(pose)
        stylecode, feats = self.style_encoding(c, retain)
        ws = self.mapping(z, stylecode, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        
        cat_feats = {}
        for i, feat in enumerate(feats):
            h, w = feat.shape[2], feat.shape[3]
            cat_feats[str(h)] = feat

        img, finetune_img, pred_parsing = self.synthesis(ws, pose_feat, cat_feats, denorm_upper_input, denorm_lower_input, \
                                                denorm_upper_mask, denorm_lower_mask, gt_parsing, **synthesis_kwargs)
        return img, finetune_img, pred_parsing

