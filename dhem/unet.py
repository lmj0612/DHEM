from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


# ----- mj_20260225: Circular 1D conv encoder for 48-dim SML/SMU condition ---------------
class CircMLTEncoder(nn.Module):
    """
    Encodes MLT-structured conditioning using a circular 1D conv over
    the 24 MLT sectors before the global MLP, giving the model structural
    knowledge of MLT adjacency and periodicity.

    For cond_dim=48: Input [B,48] -> reshape [B,2,24] -> Conv1d -> MLP
    For cond_dim=50: First 48 dims go through Conv1d, last 2 (T1,T2) are
                     concatenated after flattening before the MLP.  # mj_20260305

    Conv1d(circular): [B, n_filters, 24]
    Flatten: [B, n_filters * 24 (+ extra_dim)]
    MLP:    [B, out_dim]
    """
    def __init__(self, out_dim, n_filters=16, cond_dim=48):  # mj_20260305
        super().__init__()
        self.cond_dim = cond_dim  # mj_20260305
        self.mlt_dim = 48  # first 48 dims are MLT-structured  # mj_20260305
        self.extra_dim = cond_dim - 48  # e.g. 2 for T1/T2  # mj_20260305
        self.conv = nn.Conv1d(2, n_filters, kernel_size=3, padding=1,
                              padding_mode='circular')   # circular over MLT
        flat_dim = n_filters * 24 + self.extra_dim  # mj_20260305
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):          # x: [B, cond_dim]  # mj_20260305
        B = x.shape[0]
        x_mlt = x[:, :48].view(B, 2, 24)    # [B, 2, 24]  # mj_20260305
        h = F.relu(self.conv(x_mlt))         # [B, n_filters, 24]
        h = h.reshape(B, -1)                 # [B, n_filters * 24]
        if self.extra_dim > 0:               # mj_20260305
            h = th.cat([h, x[:, 48:]], dim=1)  # [B, n_filters*24 + extra_dim]  # mj_20260305
        return self.mlp(h)                   # [B, out_dim]
# ----- end mj_20260225 ------------------------------------------------------------------


# ----- start mj_20251020 !!!!------------------------------------------------------------
class Conv2dPeriodicWidth(nn.Module):
    """
    Circular padding only along the last dimension (MLT); reflect / zero padding along MLAT.
    kernel_size / stride / dilation as in nn.Conv2d; the inner conv uses no padding.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1,
                 pad_h=None, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        self.pad_w = (kw // 2) * dilation
        self.pad_h = (kh // 2) * dilation if pad_h is None else pad_h
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw),
                              stride=stride, padding=0, dilation=dilation, bias=bias)

    def forward(self, x):  # x: [B, C, H(MLAT), W(MLT)]
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode='circular')
        if self.pad_h and self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode='reflect')
        return self.conv(x)

def periodic_conv2d_w(in_ch, out_ch, kernel_size=3, stride=1, dilation=1, bias=True, pad_h=None):
    # only for 2D convolutions with kw > 1; 1x1 convolutions are unchanged.
    if isinstance(kernel_size, int):
        kw = kernel_size
    else:
        kw = kernel_size[1]
    if kw == 1:
        return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                         stride=stride, padding=0, dilation=dilation, bias=bias)
    return Conv2dPeriodicWidth(in_ch, out_ch, kernel_size, stride, dilation, pad_h, bias)
# ----- end mj_20251020 !!!!------------------------------------------------------------


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):  # mj_20260312: added context for cross-attn
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif getattr(layer, 'is_cross_attn', False):  # mj_20260312
                x = layer(x, context)  # mj_20260312
            elif getattr(layer, 'xattn', None) is not None:  # mj_20260927 AttentionBlock with MLT cross-attention v2 sub-module
                x = layer(x, context)  # mj_20260927
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            # self.conv = conv_nd(dims, channels, channels, 3, padding=1)
            self.conv = periodic_conv2d_w(channels, channels, kernel_size=3) # mj_20251020

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            # self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
            self.op = periodic_conv2d_w(channels, channels, kernel_size=3, stride=stride) # mj_20251020
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # ------ start mj_20251020 ------------------------------------------------------------
        if dims == 2:
            conv_in = periodic_conv2d_w(channels, self.out_channels, kernel_size=3)
        else:
            conv_in = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        # ------ end mj_20251020 ------------------------------------------------------------

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            # conv_nd(dims, channels, self.out_channels, 3, padding=1),
            conv_in, # mj_20251020
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # ------ start mj_20251020 ------------------------------------------------------------
        if dims == 2:
            conv_out = periodic_conv2d_w(self.out_channels, self.out_channels, kernel_size=3)
        else:
            conv_out = conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        # ------ end mj_20251020 ------------------------------------------------------------

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                # conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                conv_out, # mj_20251020
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            # ------ start mj_20251020 ------------------------------------------------------------
            # self.skip_connection = conv_nd(
            #     dims, channels, self.out_channels, 3, padding=1
            # )
            if dims == 2:
                self.skip_connection = periodic_conv2d_w(channels, self.out_channels, kernel_size=3)
            else:
                self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
            # ------ end mj_20251020 ------------------------------------------------------------
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.xattn = None  # mj_20260927 optional MLTCrossAttention (use_cross_attn=2), applied after the self-attention residual

    def forward(self, x, context=None):  # mj_20260927 context = [B, 24, 2] MLT-sector tokens (only used when self.xattn is set)
        x = checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)  # mj_20260927
        if getattr(self, 'xattn', None) is not None and context is not None:  # mj_20260927
            x = self.xattn(x, context)  # mj_20260927
        return x  # mj_20260927

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


# ----- mj_20260312: Cross-attention block for MLT-structured conditioning -----
class CrossAttentionBlock(nn.Module):
    """
    Cross-attention: spatial feature pixels attend to 24 MLT condition tokens.

    Q from spatial feature map [B, C, H*W]
    K, V from MLT context [B, 24, ctx_dim] -> projected to [B, C, 24]

    Each pixel at MLT≈m naturally learns to attend to nearby SML/SMU sectors.
    """

    def __init__(self, channels, ctx_dim, num_heads=1, use_checkpoint=False):  # mj_20260312
        super().__init__()
        self.channels = channels  # mj_20260312
        self.num_heads = num_heads  # mj_20260312
        self.use_checkpoint = use_checkpoint  # mj_20260312
        self.is_cross_attn = True  # mj_20260312 marker for TimestepEmbedSequential

        self.norm = normalization(channels)  # mj_20260312
        self.q_proj = conv_nd(1, channels, channels, 1)  # mj_20260312
        self.kv_proj = nn.Linear(ctx_dim, channels * 2)  # mj_20260312
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))  # mj_20260312

    def forward(self, x, context):  # mj_20260312
        return checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)

    def _forward(self, x, context):  # mj_20260312
        b, c, *spatial = x.shape  # mj_20260312
        x_flat = x.reshape(b, c, -1)  # [B, C, H*W]  # mj_20260312

        # Q from spatial features  # mj_20260312
        q = self.q_proj(self.norm(x_flat))  # [B, C, H*W]  # mj_20260312

        # K, V from MLT context  # mj_20260312
        kv = self.kv_proj(context)  # [B, 24, C*2]  # mj_20260312
        kv = kv.permute(0, 2, 1)  # [B, C*2, 24]  # mj_20260312
        k, v = th.split(kv, c, dim=1)  # each [B, C, 24]  # mj_20260312

        # Multi-head reshape  # mj_20260312
        q = q.reshape(b * self.num_heads, -1, q.shape[2])  # mj_20260312
        k = k.reshape(b * self.num_heads, -1, k.shape[2])  # mj_20260312
        v = v.reshape(b * self.num_heads, -1, v.shape[2])  # mj_20260312

        # Scaled dot-product attention  # mj_20260312
        scale = 1 / math.sqrt(math.sqrt(c // self.num_heads))  # mj_20260312
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)  # [B*H, H*W, 24]  # mj_20260312
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)  # mj_20260312
        h = th.einsum("bts,bcs->bct", weight, v)  # [B*H, C/H, H*W]  # mj_20260312

        h = h.reshape(b, -1, h.shape[-1])  # [B, C, H*W]  # mj_20260312
        h = self.proj_out(h)  # mj_20260312
        return (x_flat + h).reshape(b, c, *spatial)  # mj_20260312
# ----- end mj_20260312 -----


# ----- mj_20260927: MLT-aligned cross-attention v2 (use_cross_attn=2) -----
class MLTCrossAttention(nn.Module):
    """
    Feature-map pixels (queries) attend to the 24 MLT-sector SML/SMU tokens (keys / values).

    Differences from the mj_20260312 CrossAttentionBlock (which gave a null A/B result on cat2):
      (a) keys/values carry their MLT identity: a learnable per-sector embedding [24, 2C] is added to
          kv_proj(context). The old block had content-only keys, so the softmax could not tell WHICH
          sector a token was and the block degenerated into a content-weighted global pooling.
      (b) a learnable per-head relative-position bias table rel_bias [heads, W, 24] is added to the
          logits, initialised as a Gaussian (sigma 1.5 h, amplitude 4) in circular MLT distance between
          the pixel column (MLT = 24*j/W, same convention as mj_spatial_basis / image_datasets sector
          centres at m h) and sector m. So at step 0 a midnight column attends to the nightside sectors,
          a noon column to the dayside sectors (the paper's Sect. 3.2 description), and training may
          widen / shift / weaken that prior per head and per column.
      (c) attached as AttentionBlock.xattn instead of an extra layer in TimestepEmbedSequential ->
          no state-dict index shift (the old layout re-initialised middle_block.2 and two Upsample convs
          when warm-starting from a plain checkpoint).
    proj_out is zero-initialised -> the block is the identity at init (safe KEEP_STATE warm start).
    """

    def __init__(self, channels, width, num_heads=1, n_sectors=24, ctx_dim=2, use_checkpoint=False,
                 init_sigma_h=1.5, init_amp=4.0):  # mj_20260927 amp 4 -> ~82-84 % of the prior mass within +-2 h at W=16/8
        super().__init__()
        assert channels % num_heads == 0, (channels, num_heads)  # mj_20260927
        self.channels = channels  # mj_20260927
        self.width = int(width)  # mj_20260927 feature-map width at this resolution (16 or 8 for 32x32 input)
        self.num_heads = num_heads  # mj_20260927
        self.n_sectors = n_sectors  # mj_20260927
        self.use_checkpoint = use_checkpoint  # mj_20260927
        self.norm = normalization(channels)  # mj_20260927
        self.q_proj = conv_nd(1, channels, channels, 1)  # mj_20260927
        self.kv_proj = nn.Linear(ctx_dim, channels * 2)  # mj_20260927
        self.sector_emb = nn.Parameter(th.zeros(n_sectors, channels * 2))  # mj_20260927 learnable MLT identity of each token (K and V halves)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))  # mj_20260927
        mlt_col = 24.0 * th.arange(self.width, dtype=th.float32) / self.width  # mj_20260927 column j -> MLT 24*j/W (endpoint=False grid)
        d = (mlt_col[:, None] - th.arange(n_sectors, dtype=th.float32)[None, :]).abs()  # mj_20260927 [W, 24]
        d = th.minimum(d, 24.0 - d)  # mj_20260927 circular distance in hours
        bias0 = init_amp * th.exp(-0.5 * (d / init_sigma_h) ** 2)  # mj_20260927 Gaussian locality prior
        self.rel_bias = nn.Parameter(bias0[None].repeat(num_heads, 1, 1).contiguous())  # mj_20260927 [heads, W, 24]

    def forward(self, x, context):  # mj_20260927
        return checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)  # mj_20260927

    def _forward(self, x, context):  # mj_20260927
        b, c, h, w = x.shape  # mj_20260927
        assert w == self.width, f"MLTCrossAttention built for width {self.width}, got {w}"  # mj_20260927
        assert context.shape[1] == self.n_sectors, context.shape  # mj_20260927
        hd, dh = self.num_heads, c // self.num_heads  # mj_20260927
        x_flat = x.reshape(b, c, h * w)  # mj_20260927 [B, C, HW], pixel t = row*w + col
        q = self.q_proj(self.norm(x_flat)).reshape(b, hd, dh, h * w)  # mj_20260927 [B, hd, dh, HW]
        kv = self.kv_proj(context.to(x.dtype)) + self.sector_emb[None].to(x.dtype)  # mj_20260927 [B, 24, 2C]
        k, v = kv.chunk(2, dim=-1)  # mj_20260927 each [B, 24, C]
        k = k.reshape(b, self.n_sectors, hd, dh).permute(0, 2, 3, 1)  # mj_20260927 [B, hd, dh, 24]
        v = v.reshape(b, self.n_sectors, hd, dh).permute(0, 2, 3, 1)  # mj_20260927 [B, hd, dh, 24]
        logits = th.einsum("bhct,bhcs->bhts", q, k) * (1.0 / math.sqrt(dh))  # mj_20260927 [B, hd, HW, 24]
        bias = self.rel_bias[:, None, :, :].expand(hd, h, w, self.n_sectors).reshape(1, hd, h * w, self.n_sectors)  # mj_20260927 column-wise MLT prior
        weight = th.softmax((logits + bias.to(logits.dtype)).float(), dim=-1).type(logits.dtype)  # mj_20260927
        out = th.einsum("bhts,bhcs->bhct", weight, v).reshape(b, c, h * w)  # mj_20260927 [B, C, HW]
        return (x_flat + self.proj_out(out)).reshape(b, c, h, w)  # mj_20260927 residual, identity at init

    def attention_prior(self):  # mj_20260927 softmax of the bias alone -> [heads, W, 24] (diagnostics)
        return th.softmax(self.rel_bias.detach().float(), dim=-1)
# ----- end mj_20260927 -----


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
        use_structured_cond=False, # mj_20250529
        C_spatial=0, # mj_20250911
        use_circ_conv_cond=False,  # mj_20260225: replace flat MLP with CircMLTEncoder
        cond_dim=48,  # mj_20260301: dimension of structured condition vector (48 for SML/SMU, 5 for scalar5)
        use_cross_attn=False,  # mj_20260312: add cross-attention to MLT condition tokens; mj_20260927: 0/False none, 1/True legacy CrossAttentionBlock, 2 = MLTCrossAttention v2 (inside every AttentionBlock)
        cross_attn_ctx_dim=2,  # mj_20260312: per-token context dim (2 = SML + SMU per sector)
        image_size=32,  # mj_20260927 needed for the per-resolution MLT bias tables of MLTCrossAttention (width = image_size // ds)
        cond_drop=False,  # mj_20260927 classifier-free guidance: learnable null condition embedding (kwarg cond_null [B] bool selects it)
        xattn_heads=0,  # mj_20261026 heads of every MLTCrossAttention block: 0 = follow the parent AttentionBlock (9/27 round), >0 = fixed, independent of num_heads_upsample (lets inference switch the self-attention head count without touching the block)
        sector_drop=0.0,  # mj_20261026 training-time probability (per sample) of zeroing the 48 SML/SMU sector dims in the GLOBAL label_emb pathway only; the cross-attention context always sees them -> forces the model to route sector information through the block
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        _xa_mode = int(use_cross_attn)  # mj_20260927 0 none / 1 legacy / 2 MLT v2
        self.use_cross_attn = _xa_mode  # mj_20260312, mj_20260927 (truthy for both modes -> context is built)
        self.cross_attn_ctx_dim = cross_attn_ctx_dim  # mj_20260312
        self.image_size = image_size  # mj_20260927
        self.xattn_heads = int(xattn_heads)  # mj_20261026
        self.sector_drop = float(sector_drop)  # mj_20261026
        self.force_sector_drop = False  # mj_20261026 inference diagnostic / p=1 models: drop the global sector dims for every sample
        _xh = (lambda h: self.xattn_heads if self.xattn_heads > 0 else h)  # mj_20261026
        self.C_spatial = C_spatial # mj_20250911
        self.in_channels = in_channels + C_spatial # mj_20250911
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_structured_cond = use_structured_cond # mj_20250529
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        elif self.use_structured_cond: # mj_20250529
            if use_circ_conv_cond:  # mj_20260225: circular conv over 24 MLT sectors
                self.label_emb = CircMLTEncoder(out_dim=time_embed_dim, n_filters=16, cond_dim=cond_dim)  # mj_20260305
            else:
                self.label_emb = nn.Sequential(
                    # nn.Linear(cond_dim, hidden_dim),
                    # nn.Linear(16, time_embed_dim), # hidden_dim=time_embed_dim=4*num_channels
                    nn.Linear(cond_dim, time_embed_dim), # cond_dim: 48 for SML/SMU, 5 for scalar5  # mj_20260301
                    nn.ReLU(),
                    nn.Linear(time_embed_dim, time_embed_dim)
                ) # mj 20250527
            if cond_drop:  # mj_20260927 CFG null token: zero-init so a warm start is identical to the checkpoint when cond_null is all False
                self.null_emb = nn.Parameter(th.zeros(time_embed_dim))  # mj_20260927


        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    # conv_nd(dims, self.in_channels, model_channels, 3, padding=1)
                    periodic_conv2d_w(self.in_channels, model_channels, kernel_size=3) # mj_20251020
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    _attn = AttentionBlock(  # mj_20260927 (was appended inline)
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    if _xa_mode == 2:  # mj_20260927
                        _attn.xattn = MLTCrossAttention(ch, width=image_size // ds, num_heads=_xh(num_heads),  # mj_20260927, mj_20261026 xattn_heads
                                                        ctx_dim=cross_attn_ctx_dim, use_checkpoint=use_checkpoint)  # mj_20260927
                    layers.append(_attn)  # mj_20260927
                    if _xa_mode == 1:  # mj_20260312, mj_20260927 legacy layout only
                        layers.append(CrossAttentionBlock(  # mj_20260312
                            ch, ctx_dim=cross_attn_ctx_dim,
                            num_heads=num_heads, use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        middle_layers = [  # mj_20260312: refactored to allow cross-attn insertion
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
        ]
        if _xa_mode == 2:  # mj_20260927
            middle_layers[1].xattn = MLTCrossAttention(ch, width=image_size // ds, num_heads=_xh(num_heads),  # mj_20260927, mj_20261026
                                                       ctx_dim=cross_attn_ctx_dim, use_checkpoint=use_checkpoint)  # mj_20260927
        if _xa_mode == 1:  # mj_20260312, mj_20260927 legacy layout only
            middle_layers.append(CrossAttentionBlock(  # mj_20260312
                ch, ctx_dim=cross_attn_ctx_dim,
                num_heads=num_heads, use_checkpoint=use_checkpoint))
        middle_layers.append(  # mj_20260312
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block = TimestepEmbedSequential(*middle_layers)  # mj_20260312

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    _attn = AttentionBlock(  # mj_20260927 (was appended inline)
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    if _xa_mode == 2:  # mj_20260927
                        _attn.xattn = MLTCrossAttention(ch, width=image_size // ds, num_heads=_xh(num_heads_upsample),  # mj_20260927, mj_20261026
                                                        ctx_dim=cross_attn_ctx_dim, use_checkpoint=use_checkpoint)  # mj_20260927
                    layers.append(_attn)  # mj_20260927
                    if _xa_mode == 1:  # mj_20260312, mj_20260927 legacy layout only
                        layers.append(CrossAttentionBlock(  # mj_20260312
                            ch, ctx_dim=cross_attn_ctx_dim,
                            num_heads=num_heads_upsample, use_checkpoint=use_checkpoint))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            # zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            zero_module(periodic_conv2d_w(model_channels, out_channels, kernel_size=3)), # mj_20251020
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

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None, **kwargs): # mj_20250529
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        elif self.use_structured_cond: # mj_20250529
            cond_in = kwargs["cond"]  # mj_20261026
            if getattr(self, "sector_drop", 0.0) > 0 and (self.training or getattr(self, "force_sector_drop", False)):  # mj_20261026
                _B = cond_in.shape[0]  # mj_20261026
                if getattr(self, "force_sector_drop", False):  # mj_20261026 every sample
                    _keep = th.zeros(_B, 1, device=cond_in.device, dtype=cond_in.dtype)  # mj_20261026
                else:  # mj_20261026 per-sample Bernoulli(p) drop; 0 = z-scored mean (climatological) sector values
                    _keep = (th.rand(_B, device=cond_in.device) >= self.sector_drop).to(cond_in.dtype).view(-1, 1)  # mj_20261026
                cond_in = th.cat([cond_in[:, :48] * _keep, cond_in[:, 48:]], dim=1)  # mj_20261026 extras (Tilt, SYM-H, history, cat extras) untouched
            ce = self.label_emb(cond_in)  # mj_20260927 (was: emb = emb + label_emb(cond)), mj_20261026 cond_in
            cn = kwargs.get("cond_null", None)  # mj_20260927 [B] bool: True -> replace the condition embedding by the learned null embedding (CFG)
            if cn is not None and hasattr(self, "null_emb"):  # mj_20260927
                ce = th.where(cn.view(-1, 1), self.null_emb.view(1, -1).to(ce.dtype), ce)  # mj_20260927
            emb = emb + ce  # mj_20260927

        # ---- mj_20260312: build cross-attention context from SML/SMU ----
        context = None  # mj_20260312
        if self.use_cross_attn:  # mj_20260312
            cond_vec = kwargs["cond"]  # [B, cond_dim]  # mj_20260312
            # First 48 dims are SML[0:24] + SMU[24:48], reshape to [B, 24, 2]  # mj_20260312
            sml_smu = cond_vec[:, :48].reshape(cond_vec.shape[0], 2, 24)  # [B, 2, 24]  # mj_20260312
            context = sml_smu.permute(0, 2, 1).contiguous()  # [B, 24, 2]  # mj_20260312
        # ---- end mj_20260312 ----

        # ---- start mj_20250911 ----
        if getattr(self, "C_spatial", 0) > 0:
            cs = kwargs.get("cond_spatial", None)  # [B, C_spatial, H, W]
            assert cs is not None, "C_spatial>0 but cond_spatial not provided"

            # shape checks
            assert cs.shape[0] == x.shape[0], "batch mismatch"
            assert cs.shape[2:] == x.shape[2:], "H/W mismatch"

            # dtype/device align, then concat once
            cs = cs.to(x.dtype).to(x.device)
            x  = th.cat([x, cs], dim=1)            # -> [B, C + C_spatial, H, W]
        # ---- end mj_20250911 ----

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)  # mj_20260312: pass context
            hs.append(h)
        h = self.middle_block(h, emb, context)  # mj_20260312
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb, context)  # mj_20260312
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)

