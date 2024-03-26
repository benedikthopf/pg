from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pg_modules.blocks import DownBlock, DownBlockPatch, conv2d
from pg_modules.projector import F_RandomProj
from pg_modules.diffaug import DiffAugment

from pg_modules.crossattention import CrossAttention


class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False, slot_dim=64, heads=8, head_dim=256, use_tf=False):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        main = []
        head_xa = []

        # Head if the initial input is the full modality
        if head:
            main += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                     nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(
            DownBlock, separable=separable)

        while start_sz > end_sz:
            if nfc[start_sz // 2] <= head_dim:
                main.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            else:
                head_xa.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2

        head_xa.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))

        self.main = nn.Sequential(*main)

        self.head = nn.Sequential(*head_xa)

        self.use_tf = use_tf
        if use_tf:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=slot_dim, nhead=heads)
            self.slot_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=4)

        self.slot_dim = slot_dim

        self.cross_attention = CrossAttention(
            head_dim, do_layernorm=False, num_heads=heads, full_dropout=0)
        self.to_head_dim = nn.Linear(slot_dim, head_dim)

    def forward(self, x, c, slots=None, slot_fn=None):
        if slots is None:
            b, _, _ = x.shape
            if slot_fn is None:
                raise RuntimeError("No slots have been passed")
            else:
                slots = slot_fn(b)
                print(
                    f"Getting slots from function, mean: {slots.mean().item()}")
        # slots has shape [batch_size, num_slots, slot_dim]
        if self.use_tf:
            slots = self.slot_encoder(slots)
        # for layer, change_dim, cross_attention in zip(self.main, self.slot_to_layer_dim, self.cross_attention):

        x = self.main(x)

        b, c, w, h = x.shape

        x_xa = x.permute(0, 2, 3, 1).reshape(b, w*h, c)
        slots = self.to_head_dim(slots)
        x_xa = self.cross_attention(x_xa, slots)
        x_xa = x_xa.reshape(b, w, h, c).permute(0, 3, 1, 2)

        # out = self.head(x)
        out_xa = self.head(x_xa)

        # return torch.cat([out, out_xa], dim=1)
        return out_xa


class SingleDiscCond(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False, c_dim=1000, cmap_dim=64, embedding_dim=128):
        super().__init__()
        self.cmap_dim = cmap_dim

        # midas channels
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(
            DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)

        # additions for conditioning on class information
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
        self.embed = nn.Embedding(
            num_embeddings=c_dim, embedding_dim=embedding_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(self.embed.embedding_dim, self.cmap_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)

        # conditioning via projection
        cmap = self.embed_proj(self.embed(c.argmax(1))
                               ).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * \
            (1 / np.sqrt(self.cmap_dim))

        return out


class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=1,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        separable=False,
        patch=False,
        slot_dim=256,
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz,
                                        end_sz=8, separable=separable, patch=patch, slot_dim=slot_dim)],
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c, slots=None):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k], c, slots).view(
                features[k].size(0), -1))

        all_logits = torch.cat(all_logits, dim=1)
        return all_logits


class ProjectedDiscriminator(torch.nn.Module):
    def __init__(
        self,
        diffaug=True,
        interp224=True,
        backbone_kwargs={},
        slot_dim=256,
        **kwargs
    ):
        super().__init__()
        self.diffaug = diffaug
        self.interp224 = interp224
        self.feature_network = F_RandomProj(**backbone_kwargs)
        self.discriminator = MultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            slot_dim=slot_dim,
            **backbone_kwargs,
        )

    def train(self, mode=True):
        self.feature_network = self.feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x, c, slots=None):
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        if self.interp224:
            x = F.interpolate(x, 224, mode='bilinear', align_corners=False)

        features = self.feature_network(x)
        logits = self.discriminator(features, c, slots)

        return logits
