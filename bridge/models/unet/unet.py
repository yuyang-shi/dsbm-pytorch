from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .layers import *


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        temb_scale=1
    ):
        super().__init__()

        self.locals = [ in_channels,
                        model_channels,
                        out_channels,
                        num_res_blocks,
                        attention_resolutions,
                        dropout,
                        channel_mult,
                        conv_resample,
                        dims,
                        num_classes,
                        use_checkpoint,
                        num_heads,
                        use_scale_shift_norm,
                        resblock_updown,
                        temb_scale
                    ]
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.temb_scale = temb_scale

        # some hacky logic to allow small unets
        if self.model_channels <= 32:
            self.num_groups = 8
        else:
            self.num_groups = 32

        self.input_ch = int(channel_mult[0] * model_channels)
        ch = self.input_ch

        time_embed_dim = self.input_ch * 4
        self.time_embed = nn.Sequential(
            linear(self.input_ch, time_embed_dim),
            nn.SiLU(inplace=True),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_groups=self.num_groups
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            num_groups=self.num_groups
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_groups=self.num_groups),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_groups=self.num_groups
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            num_groups=self.num_groups
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch, self.num_groups),
            nn.SiLU(inplace=True),
            zero_module(conv_nd(dims, self.input_ch, out_channels, 3, padding=1)),
        )


    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)


    def forward(self, x, y, timesteps):

        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        timesteps = timesteps.squeeze()
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps * self.temb_scale + 1, self.input_ch))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x  # .type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, cond_channels, *args, **kwargs):
        super().__init__(in_channels + cond_channels, *args, **kwargs)
        self.locals[0] = in_channels
        self.locals.insert(1, cond_channels)

    def forward(self, x, low_res, timesteps, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, None, timesteps, **kwargs)


class DownscalerUNetModel(nn.Module):
    def __init__(
        self,
        in_channels,
        cond_channels, 
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        dims=2,
        temb_scale=1, 
        mean_bypass=False,
        scale_mean_bypass=False,
        shift_input=False,
        shift_output=False,
        **kwargs
    ):
        super().__init__()

        self.locals = [ in_channels,
                        cond_channels, 
                        model_channels,
                        out_channels,
                        num_res_blocks, 
                        dropout,
                        channel_mult,
                        dims,
                        temb_scale, 
                        mean_bypass, 
                        scale_mean_bypass, 
                        shift_input, 
                        shift_output
                    ]
        
        in_channels = in_channels + cond_channels
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.temb_scale = temb_scale

        self.mean_bypass = mean_bypass
        self.scale_mean_bypass = scale_mean_bypass
        self.shift_input = shift_input
        self.shift_output = shift_output

        assert len(channel_mult) == 4
        self.input_ch = int(channel_mult[0] * model_channels)
        ch = self.input_ch

        embed_dim = time_embed_dim = int(channel_mult[-1] * model_channels)
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            linear(time_embed_dim, time_embed_dim),
            nn.SiLU(inplace=True),
            # linear(time_embed_dim, time_embed_dim),
        )

        if self.mean_bypass:
            self.mean_skip_1 = conv_nd(dims, in_channels, embed_dim, 1)  # Conv((1, 1), inchannels => embed_dim)
            self.mean_skip_2 = conv_nd(dims, embed_dim, embed_dim, 1)  # Conv((1, 1), embed_dim => embed_dim)
            self.mean_skip_3 = conv_nd(dims, embed_dim, out_channels, 1)  # Conv((1, 1), embed_dim => outchannels)
            self.mean_dense_1 = linear(embed_dim, embed_dim)
            self.mean_dense_2 = linear(embed_dim, embed_dim)
            self.mean_gnorm_1 = normalization_act(embed_dim, 32)  # GroupNorm(embed_dim, 32, swish)
            self.mean_gnorm_2 = normalization_act(embed_dim, 32)  # GroupNorm(embed_dim, 32, swish)

        self.conv1 = conv_nd(dims, in_channels, ch, 3, padding=1)  # 3 -> 32  # Conv((3, 3), inchannels => channels[1], stride=1, pad=SamePad())
        self.dense1 = linear(time_embed_dim, ch)  # Dense(embed_dim, channels[1]),
        self.gnorm1 = normalization_act(ch, 4)  # GroupNorm(channels[1], 4, swish),

        # Encoding
        out_ch = int(channel_mult[1] * model_channels)
        self.conv2 = Downsample(ch, use_conv=True, dims=dims, out_channels=out_ch)  # 32 -> 64
        self.dense2 = linear(time_embed_dim, out_ch)
        self.gnorm2 = normalization_act(out_ch, 32)

        ch = out_ch
        out_ch = int(channel_mult[2] * model_channels)
        self.conv3 = Downsample(ch, use_conv=True, dims=dims, out_channels=out_ch)  # 64 -> 128
        self.dense3 = linear(time_embed_dim, out_ch)
        self.gnorm3 = normalization_act(out_ch, 32)
        
        ch = out_ch
        out_ch = int(channel_mult[3] * model_channels)
        self.conv4 = Downsample(ch, use_conv=True, dims=dims, out_channels=out_ch)  # 128 -> 256
        self.dense4 = linear(time_embed_dim, out_ch)

        self.middle_block = TimestepEmbedSequential(
            *[
                ResBlock(
                    out_ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    num_groups=min(out_ch//4, 32)
                ) for _ in range(num_res_blocks)
            ]
        )

        # Decoding
        self.gnorm4 = normalization_act(out_ch, 32)
        self.tconv4 = Upsample(out_ch, use_conv=True, dims=dims, out_channels=ch)  # 256 -> 128
        self.denset4 = linear(time_embed_dim, ch)
        self.tgnorm4 = normalization_act(ch, 32)
        
        out_ch = ch
        ch = int(channel_mult[1] * model_channels)
        self.tconv3 = Upsample(out_ch*2, use_conv=True, dims=dims, out_channels=ch)  # 128 + 128 -> 64
        self.denset3 = linear(time_embed_dim, ch)
        self.tgnorm3 = normalization_act(ch, 32)

        out_ch = ch
        ch = int(channel_mult[0] * model_channels)
        self.tconv2 = Upsample(out_ch*2, use_conv=True, dims=dims, out_channels=ch)  # 64 + 64 -> 32
        self.denset2 = linear(time_embed_dim, ch)
        self.tgnorm2 = normalization_act(ch, 32)

        self.tconv1 = zero_module(conv_nd(dims, self.input_ch*2, out_channels, 3, padding=1))


    def forward(self, x, y, timesteps):
        timesteps = timesteps.squeeze()
        embed = self.time_embed(timestep_embedding(timesteps * self.temb_scale + 1, self.time_embed_dim))

        # Encoder
        if self.shift_input:
            h1 = x - th.mean(x, dim=(-1,-2), keepdim=True) # remove mean of noised variables before input
        else:
            h1 = x

        h1 = th.cat([x, y], dim=1)
        h1 = self.conv1(h1)
        h1 = h1 + expand_dims(self.dense1(embed), len(h1.shape))
        h1 = self.gnorm1(h1)
        h2 = self.conv2(h1)
        h2 = h2 + expand_dims(self.dense2(embed), len(h2.shape))
        h2 = self.gnorm2(h2)
        h3 = self.conv3(h2)
        h3 = h3 + expand_dims(self.dense3(embed), len(h3.shape))
        h3 = self.gnorm3(h3)
        h4 = self.conv4(h3)
        h4 = h4 + expand_dims(self.dense4(embed), len(h4.shape))

        # middle
        h = h4
        h = self.middle_block(h, embed)

        # Decoder
        h = self.gnorm4(h)
        h = self.tconv4(h)
        h = h + expand_dims(self.denset4(embed), len(h.shape))
        h = self.tgnorm4(h)
        h = self.tconv3(th.cat([h, h3], dim=1))
        h = h + expand_dims(self.denset3(embed), len(h.shape))
        h = self.tgnorm3(h)
        h = self.tconv2(th.cat([h, h2], dim=1))
        h = h + expand_dims(self.denset2(embed), len(h.shape))
        h = self.tgnorm2(h)
        h = self.tconv1(th.cat([h, h1], dim=1))

        if self.shift_output:
            h = h - th.mean(h, dim=(-1,-2), keepdim=True) # remove mean after output

        # Mean processing of noised variable channels
        if self.mean_bypass:
            hm = self.mean_skip_1(th.mean(th.cat([x, y], dim=1), dim=(-1,-2), keepdim=True))
            hm = hm + expand_dims(self.mean_dense_1(embed), len(hm.shape))
            hm = self.mean_gnorm_1(hm)
            hm = self.mean_skip_2(hm)
            hm = hm + expand_dims(self.mean_dense_2(embed), len(hm.shape))
            hm = self.mean_gnorm_2(hm)
            hm = self.mean_skip_3(hm)
            if self.scale_mean_bypass:
                scale = np.sqrt(np.prod(x.shape[2:]))
                hm = hm / scale
            # Add back in noised channel mean to noised channel spatial variatons
            return h + hm
        else:
            return h
