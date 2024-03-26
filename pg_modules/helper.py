import numpy as np
from torch import nn
import torch
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DinoAutoencoder(nn.Module):
    def __init__(self, encoder_name="vit_small_patch16_224_dino", decoder_dim=64):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=True)
        for parameter in self.encoder.parameters():  # freeze encoder
            parameter.requires_grad = False

        self.n_patches = 196 if "patch16" in encoder_name else 784
        self.feature_map_shape = int(self.n_patches**0.5)
        self.d_feat = 384 if "small" in encoder_name else 768

        self.decoder = [
            nn.Conv2d(self.d_feat, self.d_feat, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.d_feat, decoder_dim, 3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
            nn.ReLU()
        ]
        if self.n_patches == 196:
            self.decoder.append(nn.ConvTranspose2d(
                decoder_dim, decoder_dim, 3, stride=2, padding=1, output_padding=1))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.ConvTranspose2d(
            decoder_dim, 3, 3, stride=2, padding=1, output_padding=1))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        b, c, w, h = x.shape
        # x has shape [batch_size, channels, width, height]
        features = self.encode(x)
        # features has shape [batch_size, num_patches, d_feat]
        features = features.permute(0, 2, 1) \
            .reshape(b, self.d_feat, self.feature_map_shape, self.feature_map_shape)
        # features has shape [batch_size, d_feat, sqrt(num_patches), sqrt(num_patches)]
        x = self.decode(features)
        # x has shape [bazch_size, 3, width, height]

        return x

    def encode(self, x):
        # x has shape [batch_size, channels, width, height]
        features = self.encoder.forward_features(x)
        # features has shape [batch_size, num_patches+1, d_feat]
        features = features[:, 1:]  # remove class token
        # features has shape [batch_size, num_patches, d_feat]
        return features

    def decode(self, features):
        return self.decoder(features)
