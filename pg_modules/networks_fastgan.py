# original implementation: https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
# modified by Benedikt Hopf for "Object-centric GAN"
#
import torch.nn as nn
from pg_modules.blocks import (
    InitLayer, UpBlockBig, UpBlockBigCond, UpBlockSmall, UpBlockSmallCond, SEBlock, conv2d)
import torch

from pg_modules.crossattention import CrossAttention


def normalize_second_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class DummyMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, c, **kwargs):
        return z.unsqueeze(1)  # to fit the StyleGAN API


class FastganSynthesis(nn.Module):
    def __init__(self, ngf=128, z_dim=256, nc=3, img_resolution=256, lite=False, slot_dim=64, heads=8, xa_features="8+16+32", use_tf=False):
        super().__init__()
        self.img_resolution = img_resolution
        self.z_dim = z_dim

        # channel multiplier
        nfc_multi = {2: 16, 4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5,
                     512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        # layers
        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)

        UpBlock = UpBlockSmall if lite else UpBlockBig

        self.feat_8 = UpBlock(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlock(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlock(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

        self.use_tf = use_tf
        if use_tf:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=slot_dim, nhead=heads)
            self.slot_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=4)

        self.slot_dim = slot_dim

        self.xa_features = xa_features

        if "8" in xa_features:
            self.slots_to_feat_8 = nn.Linear(slot_dim, nfc[8])

            self.cross_attention_8 = CrossAttention(
                nfc[8], num_heads=8, do_layernorm=False)

        if "16" in xa_features:
            self.slots_to_feat_16 = nn.Linear(slot_dim, nfc[16])
            self.cross_attention_16 = CrossAttention(
                nfc[16], num_heads=8, do_layernorm=False)

        if "32" in xa_features:
            self.slots_to_feat_32 = nn.Linear(slot_dim, nfc[32])
            self.cross_attention_32 = CrossAttention(
                nfc[32], num_heads=8, do_layernorm=False)

        # #############################
        # self.pos_embed = nn.Transformer(
        #     slot_dim,
        #     8,
        #     num_encoder_layers=4,
        #     num_decoder_layers=3,
        #     dim_feedforward=512,
        #     batch_first=True
        # )
        # self.to_pos_embed_dim = nn.Linear(32, slot_dim)

    def forward(self, input, c, slots=None, slot_fn=None, **kwargs):
        if slots is None:
            b, _, _ = input.shape
            if slot_fn is None:
                raise RuntimeError("No slots have been passed")
            else:
                slots = slot_fn(b)

        # map noise to hypersphere as in "Progressive Growing of GANS"
        input = normalize_second_moment(input[:, 0])

        # ######################## Position learning ###################
        # pos_base = input[:, :7*32].reshape(-1, 7, 32)
        # pos_base = self.to_pos_embed_dim(pos_base)
        # slots = self.pos_embed(pos_base, slots)
        # ########################

        if self.use_tf:
            slots = self.slot_encoder(slots)

        feat_4 = self.init(input)

        feat_8 = self.feat_8(feat_4)
        if "8" in self.xa_features:
            s = self.slots_to_feat_8(slots)
            b, c, w, h, = feat_8.shape
            feat_8 = feat_8.permute(0, 2, 3, 1).reshape(b, w*h, c)
            feat_8 = self.cross_attention_8(feat_8, s)
            feat_8 = feat_8.reshape(b, w, h, c).permute(0, 3, 1, 2)

        feat_16 = self.feat_16(feat_8)
        if "16" in self.xa_features:
            s = self.slots_to_feat_16(slots)
            b, c, w, h, = feat_16.shape
            feat_16 = feat_16.permute(0, 2, 3, 1).reshape(b, w*h, c)
            feat_16 = self.cross_attention_16(feat_16, s)
            feat_16 = feat_16.reshape(b, w, h, c).permute(0, 3, 1, 2)

        feat_32 = self.feat_32(feat_16)
        if "32" in self.xa_features:
            s = self.slots_to_feat_32(slots)
            b, c, w, h, = feat_32.shape
            feat_32 = feat_32.permute(0, 2, 3, 1).reshape(b, w*h, c)
            feat_32 = self.cross_attention_32(feat_32, s)
            feat_32 = feat_32.reshape(b, w, h, c).permute(0, 3, 1, 2)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
        feat_128 = self.se_128(feat_8,  self.feat_128(feat_64))

        if self.img_resolution >= 128:
            feat_last = feat_128

        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last))

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last))

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last)

        return self.to_big(feat_last)


class FastganSynthesisCond(nn.Module):
    def __init__(self, ngf=64, z_dim=256, nc=3, img_resolution=256, num_classes=1000, lite=False):
        super().__init__()

        self.z_dim = z_dim
        nfc_multi = {2: 16, 4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5,
                     512: 0.25, 1024: 0.125, 2048: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.img_resolution = img_resolution

        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)

        UpBlock = UpBlockSmallCond if lite else UpBlockBigCond

        self.feat_8 = UpBlock(nfc[4], nfc[8], z_dim)
        self.feat_16 = UpBlock(nfc[8], nfc[16], z_dim)
        self.feat_32 = UpBlock(nfc[16], nfc[32], z_dim)
        self.feat_64 = UpBlock(nfc[32], nfc[64], z_dim)
        self.feat_128 = UpBlock(nfc[64], nfc[128], z_dim)
        self.feat_256 = UpBlock(nfc[128], nfc[256], z_dim)

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

        self.embed = nn.Embedding(num_classes, z_dim)

    def forward(self, input, c, update_emas=False):
        c = self.embed(c.argmax(1))

        # map noise to hypersphere as in "Progressive Growing of GANS"
        input = normalize_second_moment(input[:, 0])

        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4, c)
        feat_16 = self.feat_16(feat_8, c)
        feat_32 = self.feat_32(feat_16, c)
        feat_64 = self.se_64(feat_4, self.feat_64(feat_32, c))
        feat_128 = self.se_128(feat_8,  self.feat_128(feat_64, c))

        if self.img_resolution >= 128:
            feat_last = feat_128

        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last, c))

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last, c))

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last, c)

        return self.to_big(feat_last)


class Generator(nn.Module):
    def __init__(
        self,
        z_dim=256,
        c_dim=0,
        w_dim=0,
        img_resolution=256,
        img_channels=3,
        ngf=128,
        cond=0,
        slot_dim=256,
        mapping_kwargs={},
        synthesis_kwargs={}
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        # Mapping and Synthesis Networks
        self.mapping = DummyMapping()  # to fit the StyleGAN API
        Synthesis = FastganSynthesisCond if cond else FastganSynthesis
        self.synthesis = Synthesis(
            ngf=ngf, z_dim=z_dim, nc=img_channels, img_resolution=img_resolution, slot_dim=slot_dim, **synthesis_kwargs)

    def forward(self, z, c, slots=None, **kwargs):
        w = self.mapping(z, c)
        img = self.synthesis(w, c, slots, **kwargs)
        return img
