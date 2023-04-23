import torch.nn as nn
import math
from .base import EqualLinear, ConvLayer, ResBlock, StyledResBlock, ReferenceAttention


class Mapping(nn.Module):
    def __init__(self, input_dim=256, latent_dim=256, num_layers=4):
        super(Mapping, self).__init__()

        layer = [EqualLinear(input_dim, latent_dim)]
        for i in range(num_layers - 1):
            layer.append(EqualLinear(latent_dim, latent_dim))
        self.layers = nn.Sequential(*layer)

    def forward(self, z):
        return self.layers(z)


class Generator(nn.Module):
    def __init__(
            self,
            channel,
            vector_dim=256,
            factor_dim=2048,
            blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        self.mlp = nn.Linear(vector_dim, 256)

        stem = [ConvLayer(1, channel, 1, blur_kernel=blur_kernel),
                ResBlock(channel, channel * 2, downsample=False, padding="reflect", blur_kernel=blur_kernel),
                ResBlock(channel * 2, channel * 4, downsample=False, padding="reflect", blur_kernel=blur_kernel),
                ResBlock(channel * 4, channel * 4, downsample=False, padding="reflect", blur_kernel=blur_kernel),
                ConvLayer(channel * 4, channel * 4, 1, blur_kernel=blur_kernel)]

        self.structure = nn.Sequential(*stem)

        ch_multiplier = (4, 8, 12, 16, 16, 16, 8, 4)
        upsample = (False, False, False, False, True, True, True, True)

        self.layers = nn.ModuleList()
        in_ch = channel * 4
        for ch_mul, up in zip(ch_multiplier, upsample):
            self.layers.append(
                StyledResBlock(
                    in_ch, channel * ch_mul, factor_dim, up, blur_kernel
                )
            )
            in_ch = channel * ch_mul

        self.to_rgb = ConvLayer(in_ch, 3, 1, activate=False)

    def forward(self, z, style):

        out = self.mlp(z).view(z.shape[0], -1, 16, 16)
        out = self.structure(out)

        for layer in self.layers:
            out = layer(out, style, None)

        out = self.to_rgb(out)
        return out


class FeatEncoder(nn.Module):
    def __init__(
            self,
            channel,
            blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        conv = [ConvLayer(3, channel, 1)]

        in_channel = channel
        for i in range(1, 5):
            ch = channel * (2 ** i)
            conv.append(ResBlock(in_channel, ch, downsample=True, padding="reflect", blur_kernel=blur_kernel))
            in_channel = ch

        self.conv = nn.Sequential(*conv)

        self.structure = nn.Sequential(
            ConvLayer(in_channel, in_channel, 1, blur_kernel=blur_kernel),
            ConvLayer(in_channel, channel * 4, 1, blur_kernel=blur_kernel))

    def forward(self, input):
        out = self.conv(input)
        structure = self.structure(out)
        return structure


class Predictor(nn.Module):
    def __init__(
            self,
            channel,
            vector_dim=256,
            blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        self.siamese = FeatEncoder(channel)
        self.reference_attention = ReferenceAttention(channel * 4)

        self.conv = nn.Sequential(
            ConvLayer(channel * 4, channel * 4, 1, blur_kernel=blur_kernel),
            ResBlock(channel * 4, channel * 4, downsample=False, padding="reflect", blur_kernel=blur_kernel),
            ResBlock(channel * 4, channel * 2, downsample=False, padding="reflect", blur_kernel=blur_kernel),
            ResBlock(channel * 2, channel, downsample=False, padding="reflect", blur_kernel=blur_kernel),
            ConvLayer(channel, 1, 1, blur_kernel=blur_kernel))

        self.mlp = nn.Linear(256, vector_dim)

    # input_image = reconstructed (or rand) image; target_image = original (container) image
    def forward(self, input_image, target_image):
        input_feat = self.siamese(input_image)
        target_feat = self.siamese(target_image)

        out = self.reference_attention(input_feat, target_feat)

        out = self.conv(out)
        out = self.mlp(out.view(out.shape[0], -1))

        return out
