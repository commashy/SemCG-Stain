class UNetModel(nn.Module):
    """
    ## U-Net model
    """

    def __init__(
            self, *,
            in_channels: int,
            out_channels: int,
            channels: int,
            n_res_blocks: int,
            attention_levels: List[int],
            channel_multipliers: List[int],
            n_heads: int,
            tf_layers: int = 1,
            d_cond: int = 768):
        """
        :param in_channels: is the number of channels in the input feature map
        :param out_channels: is the number of channels in the output feature map
        :param channels: is the base channel count for the model
        :param n_res_blocks: number of residual blocks at each level
        :param attention_levels: are the levels at which attention should be performed
        :param channel_multipliers: are the multiplicative factors for number of channels for each level
        :param n_heads: is the number of attention heads in the transformers
        :param tf_layers: is the number of transformer layers in the transformers
        :param d_cond: is the size of the conditional embedding in the transformers
        """
        super().__init__()
        self.channels = channels

        # Number of levels
        levels = len(channel_multipliers)

        # patch embedding for condtion generation
        self.patch_embed = PatchEmbedding()

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()

        # First convolution
        self.input_blocks.append(TimestepEmbedSequential(
            nn.Conv2d(in_channels, channels, 3, padding=1)))
        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]
        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(n_res_blocks):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                layers = [ResBlock(channels, out_channels=channels_list[i])]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                # Add them to the input half of the U-Net and keep track of the number of channels of
                # its output
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels),
            SpatialTransformer(channels, n_heads, tf_layers, d_cond),
            ResBlock(channels),
        )

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            # Add the residual blocks and attentions
            for j in range(n_res_blocks + 1):
                # Residual block maps from previous number of channels plus the
                # skip connections from the input half of U-Net to the number of
                # channels in the current level.
                layers = [ResBlock(channels + input_block_channels.pop(), out_channels=channels_list[i])]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                # Up-sample at every level after last residual block
                # except the last one.
                # Note that we are iterating in reverse; i.e. `i == 0` is the last.
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))
                # Add to the output half of the U-Net
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Final normalization and $3 \times 3$ convolution
        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def forward(self, x_large: torch.Tensor, full=False):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """
         # Step 1: Generate image embedding from the larger input
        if full:
            cond = self.patch_embed(x_large)
        else:
            fake_x_large = F.interpolate(x_large, size=(768, 768), mode='bilinear', align_corners=False)
            cond = self.patch_embed(fake_x_large)

        # Step 2: Extract the center 256x256 patch from the larger 768x768 input
        if full:
            center_size = 256
            large_size = x_large.shape[2]
            start = (large_size - center_size) // 2
            end = start + center_size
            x = x_large[:, :, start:end, start:end]
        else:
            x = x_large

        # To store the input half outputs for skip connections
        x_input_block = []

        # Input half of the U-Net
        for module in self.input_blocks:
            x = module(x, cond)
            x_input_block.append(x)
        # Middle of the U-Net
        x = self.middle_block(x, cond)
        # Output half of the U-Net
        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, cond)

        # Final normalization and $3 \times 3$ convolution
        return self.out(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, patch_size=88, stride=88, num_patches=77):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))  # Learnable positional embeddings

    def forward(self, x):
        # Apply the patch embedding (conv layer) with overlap
        x = self.proj(x)  # [batch_size, embed_dim, H/patch_size, W/patch_size]
        
        # Flatten the patches and permute dimensions to get [batch_size, num_patches_actual, embed_dim]
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches_actual, embed_dim]
        
        return x


class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs

    This sequential module can compose of different modules such as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class ResBlock(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(self, channels: int, out_channels=None):
        """
        :param channels: the number of input channels
        :param d_t_emb: the size of timestep embeddings
        :param out_channels: is the number of out channels. defaults to `channels.
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        # Final convolution layer
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        # Initial convolution
        h = self.in_layers(x)
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h


class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)

class InstanceNorm(nn.InstanceNorm2d):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups..
    """
    return InstanceNorm(channels)

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SpatialTransformer(nn.Module):
    """
    ## Spatial Transformer
    """

    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
        """
        :param channels: is the number of channels in the feature map
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of transformer layers
        :param d_cond: is the size of the conditional embedding
        """
        super().__init__()
        # Initial group normalization
        # self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.norm = nn.InstanceNorm2d(channels)
        # Initial $1 \times 1$ convolution
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_heads, channels // n_heads, d_cond=d_cond) for _ in range(n_layers)]
        )

        # Final $1 \times 1$ convolution
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: is the feature map of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        # Get shape `[batch_size, channels, height, width]`
        b, c, h, w = x.shape
        # For residual connection
        x_in = x
        # Normalize
        x = self.norm(x)
        # Initial $1 \times 1$ convolution
        x = self.proj_in(x)
        # Transpose and reshape from `[batch_size, channels, height, width]`
        # to `[batch_size, height * width, channels]`
        x = x.permute(0, 2, 3, 1).view(b, h * w, c)
        # Apply the transformer layers
        for block in self.transformer_blocks:
            x = block(x, cond)
        # Reshape and transpose from `[batch_size, height * width, channels]`
        # to `[batch_size, channels, height, width]`
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        # Final $1 \times 1$ convolution
        x = self.proj_out(x)
        # Add residual
        return x + x_in


class BasicTransformerBlock(nn.Module):
    """
    ### Transformer Layer
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        """
        super().__init__()
        # Self-attention layer and pre-norm layer
        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        # Cross attention layer and pre-norm layer
        self.attn2 = CrossAttention(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)
        # Feed-forward network and pre-norm layer
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        # Self attention
        x = self.attn1(self.norm1(x)) + x
        # Cross-attention with conditioning
        x = self.attn2(self.norm2(x), cond=cond) + x
        # Feed-forward network
        x = self.ff(self.norm3(x)) + x
        #
        return x


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer

    This falls-back to self-attention when conditional embeddings are not specified.
    """

    use_flash_attention: bool = False

    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool = True):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head

        # Attention scaling factor
        self.scale = d_head ** -0.5

        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond, d_attn, bias=False)

        # Final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))

        # Setup [flash attention](https://github.com/HazyResearch/flash-attention).
        # Flash attention is only used if it's installed
        # and `CrossAttention.use_flash_attention` is set to `True`.
        try:
            # You can install flash attention by cloning their Github repo,
            # [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
            # and then running `python setup.py install`
            from flash_attn.flash_attention import FlashAttention
            self.flash = FlashAttention()
            # Set the scale for scaled dot-product attention.
            self.flash.softmax_scale = self.scale
        # Set to `None` if it's not installed
        except ImportError:
            self.flash = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """

        # If `cond` is `None` we perform self attention
        has_cond = cond is not None
        if not has_cond:
            cond = x

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        # Use flash attention if it's available and the head size is less than or equal to `128`
        if CrossAttention.use_flash_attention and self.flash is not None and not has_cond and self.d_head <= 128:
            return self.flash_attention(q, k, v)
        # Otherwise, fallback to normal attention
        else:
            return self.normal_attention(q, k, v)

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Flash Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Get batch size and number of elements along sequence axis (`width * height`)
        batch_size, seq_len, _ = q.shape

        # Stack `q`, `k`, `v` vectors for flash attention, to get a single tensor of
        # shape `[batch_size, seq_len, 3, n_heads * d_head]`
        qkv = torch.stack((q, k, v), dim=2)
        # Split the heads
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        # Flash attention works for head sizes `32`, `64` and `128`, so we have to pad the heads to
        # fit this size.
        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f'Head size ${self.d_head} too large for Flash Attention')

        # Pad the heads
        if pad:
            qkv = torch.cat((qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim=-1)

        # Compute attention
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        # This gives a tensor of shape `[batch_size, seq_len, n_heads, d_padded]`
        out, _ = self.flash(qkv)
        # Truncate the extra head size
        out = out[:, :, :, :self.d_head]
        # Reshape to `[batch_size, seq_len, n_heads * d_head]`
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)

        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention
        
        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        # Compute softmax
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$$
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # Compute attention output
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, height * width, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    ### Feed-Forward Network
    """

    def __init__(self, d_model: int, d_mult: int = 4):
        """
        :param d_model: is the input embedding size
        :param d_mult: is multiplicative factor for the hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(0.),
            nn.Linear(d_model * d_mult, d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * F.gelu(gate)