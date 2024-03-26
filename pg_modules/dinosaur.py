##### Based on https://arxiv.org/abs/2209.14860 #######


import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import timm
from pg_modules.slot_attention import SlotAttention, SoftPositionEmbed
from sklearn.decomposition import PCA
from pg_modules.slate_transformer import TransformerDecoder
from torchvision.transforms import Resize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""Implementation of the DINOSAUR Module """


class DINOSAUR(nn.Module):
    def __init__(self, num_slots, num_iterations, slot_dim, decoder_dim, encoder_name="vit_base_patch8_224_dino", use_transformer=False):
        super().__init__()

        if "base" in encoder_name:
            self.d_feat = 768  # number of features
        else:
            self.d_feat = 384
        if "patch8" in encoder_name:
            self.N = 784  # number of patches
            self.mask_res = (28, 28)
        else:
            self.N = 196  # number of patches
            self.mask_res = (14, 14)

        self.encoder = timm.create_model(encoder_name, pretrained=True)

        for parameter in self.encoder.parameters():  # freeze encoder
            parameter.requires_grad = False

        self.encoder_to_sa = nn.Sequential(
            nn.Linear(self.d_feat, self.d_feat),
            nn.ReLU(),
            nn.Linear(self.d_feat, slot_dim)
        )
        self.slot_attention = SlotAttention(
            num_slots, slot_dim, num_iterations, hidden_dim=4*slot_dim, return_masks=True)
        self.pos_embed = SoftPositionEmbed(slot_dim, self.mask_res)

        if not use_transformer:
            self.decoder = nn.Sequential(
                nn.Linear(slot_dim, decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, self.d_feat+1),
            )
            self.softmax_mask = nn.Softmax(dim=1)
            self.decode_fn = self.decode

        else:
            self.slots_to_decoder = nn.Linear(slot_dim, self.d_feat)
            self.features_to_decoder = nn.Linear(self.d_feat+1, self.d_feat)
            self.tf_dec = TransformerDecoder(
                num_blocks=4,
                max_len=self.N,
                d_model=self.d_feat,
                num_heads=3,
                dropout=0.0
            )

            self.out = nn.Linear(self.d_feat, self.d_feat, bias=False)
            self.decode_fn = lambda *args, **kwargs: self.decode_transformer(
                *args, **kwargs) if not self.autoregressive else self.decode_autoregressive(*args, **kwargs)

        self.layernorm_features = nn.LayerNorm((self.N, self.d_feat))
        self.layernorm_before_slots = nn.LayerNorm((self.N, slot_dim))

        self.PCA = PCA
        self.PCA_dataset = []

        self.autoregressive = False

    def forward(self, x):
        # x has shape [batch_size, channels, width, height] = [batch_size, 3, 224, 224]
        features, x = self.encode(x)
        # features has shape [batch_size, num_patches, d_feat] = [batch_size, 784, 768]
        # x has shape [batch_size, num_patches, slot_dim]

        slots, masks = self.slot_attention(x)
        # slots has shape [batch_size, num_slots, slot_dim]

        recon_combined, recons, masks = self.decode_fn(slots, features, masks)
        # recons has shape [batch_size, num_slots, num_patches, d_feat]
        # masks has shape [batch_size, num_slots, num_patches, 1]
        # recon_combined has shape [batch_size, num_patches, d_feat]

        return features, recon_combined, recons, masks, slots

    def encode(self, x):
        # x has shape [batch_size, channels, width, height] = [batch_size, 3, 224, 224]
        features = self.encoder.forward_features(x)
        # features has shape [batch_size, num_patches+1, d_feat] = [batch_size, 785, 768]
        features = features[:, 1:]  # remove class token
        # features has shape [batch_size, num_patches, d_feat] = [batch_size, 784, 768]
        x = self.layernorm_features(features)
        x = self.encoder_to_sa(x)
        # x has shape [batch_size, num_patches, slot_dim]
        x = self.layernorm_before_slots(x)

        return features, x

    def decode(self, slots, features, masks):
        x = slots.unsqueeze(2).repeat(1, 1, features.shape[1], 1)
        # x has shape [batch_size, num_slots, num_patches, slot_dim]
        b, n_s, n_p, d = x.shape
        wh = int(n_p**0.5)
        x = x.reshape(b*n_s, wh, wh, d)  # reshape to use 2d position embedding
        # x has shape [batch_size*num_slots, width=sqrt(num_patches), height=sqrt(num_patches), slot_dim]
        x = self.pos_embed(x)
        # x has shape [batch_size*num_slots, width=sqrt(num_patches), height=sqrt(num_patches), slot_dim]
        x = x.reshape(b, n_s, n_p, d)
        # x has shape [batch_size, num_slots, num_patches, slot_dim]
        x = self.decoder(x)
        # x has shape [batch_size, num_slots, num_patches, d_feat+1]

        recons, masks = x.split([self.d_feat, 1], dim=-1)
        masks = self.softmax_mask(masks)
        # recons has shape [batch_size, num_slots, num_patches, d_feat]
        # masks has shape [batch_size, num_slots, num_patches, 1]
        recon_combined = torch.sum(recons*masks, dim=1)
        # recon_combined has shape [batch_size, num_patches, d_feat]

        return recon_combined, recons, masks

    def decode_transformer(self, slots, features, masks):
        # features has shape [batch_size, num_patches, d_feat] = [batch_size, 784, 768]

        # slots has shape [batch_size, num_slots, slot_dim]
        slots = self.slots_to_decoder(slots)
        # slots has shape [batch_size, num_slots, d_feat]
        f = torch.cat(
            [torch.zeros_like(features[..., :1, :]), features], dim=-2)
        f = torch.cat([torch.zeros_like(f[..., :1]), f], dim=-1)
        f[..., 0, 0] = 1
        # f has dim [batch_size, num_patches + 1, d_feat+1]
        f = self.features_to_decoder(f)
        # f has dim [batch_size, num_patches + 1, d_feat]

        recon_combined = self.tf_dec(f[:, :-1], slots)
        # recon_combined has shape [batch_size, num_patches, d_feat]
        b, n, d = recon_combined.shape
        _, s, _ = masks.shape
        # masks has shape [batch_size, num_slots, num_patches]
        recons = (recon_combined.reshape(b, 1, n, d)
                  * masks.reshape(b, s, n, 1))
        # recons has shape [batch_size, num_slots, num_patches, d_feat]
        masks = masks.reshape(b, s, n, 1)
        # masks has shape [batch_size, num_slots, num_patches, 1]

        return recon_combined, recons, masks

    def decode_autoregressive(self, slots, features, masks):

        with torch.no_grad():
            # features are only used for dimensions and to support same signature as decode_transformer:
            if type(features) == type(torch.ones(1)):
                features_shape = features.shape
            else:
                features_shape = features

            # feature_shape is [batch_size, num_patches, d_feat]
            # slots has shape [batch_size, num_slots, slot_dim]
            slots = self.slots_to_decoder(slots)
            # slots has shape [batch_size, num_slots, d_feat]

            b, s, d = slots.shape
            b, n, d = features_shape

            features_bos = torch.zeros(b, 1, d+1).to(device)
            features_bos[:, 0, 0] = 1  # BOS

            recon_combined = None

            self.mses = []

            for t in range(n):

                transformer_input = self.features_to_decoder(features_bos)

                new_output = self.tf_dec(transformer_input, slots)
                next_patch = new_output[:, -1:]

                self.mses.append(nn.MSELoss()(
                    next_patch, features[:, t:t+1]).detach().cpu().numpy())

                features_bos = torch.cat([
                    features_bos,
                    torch.cat([
                        torch.zeros_like(next_patch[:, :, :1]),
                        next_patch  # features[:, t:t+1],
                    ],
                        dim=-1
                    )
                ], dim=1
                )

                if recon_combined is None:
                    recon_combined = next_patch
                else:
                    recon_combined = torch.cat(
                        [recon_combined, next_patch], dim=1)

            # recon_combined has shape [batch_size, num_patches, d_feat]
            b, n, d = recon_combined.shape
            _, s, _ = masks.shape
            # masks has shape [batch_size, num_slots, num_patches]
            recons = (recon_combined.reshape(b, 1, n, d)
                      * masks.reshape(b, s, n, 1))
            # recons has shape [batch_size, num_slots, num_patches, d_feat]
            masks = masks.reshape(b, s, n, 1)
            # masks has shape [batch_size, num_slots, num_patches, 1]

            return recon_combined, recons, masks

    def clear_PCA_dataset(self):
        self.PCA_dataset = []

    def add_to_PCA_dataset(self, x, label_fn=lambda i, m: "unknown"):
        """Add pair of slot_i(x), label_i for i \in num_slots to the PCA dataset
            if label == "ignored", the pair will not be added 
        """
        for image in x.unsqueeze(1):  # manual batch size of 1, in order to use the same seed everytime
            torch.manual_seed(1)
            # x has shape [1, channels, width, height] = [batch_size, 3, 224, 224]
            features, x = self.encode(image)
            # features has shape [1, num_patches, d_feat] = [batch_size, 784, 768]
            # x has shape [1, num_patches, slot_dim]

            slots, masks = self.slot_attention(x)
            # slots has shape [1, num_slots, slot_dim]

            recon_combined, recons, masks = self.decode_fn(
                slots, features, masks)
            # recons has shape [1, num_slots, num_patches, d_feat]
            # masks has shape [1, num_slots, num_patches, 1]
            # recon_combined has shape [1, num_patches, d_feat]

            b, n, d = slots.shape

            slots = slots.reshape(-1, d)
            # slots has shape [num_slots, slot_dim]

            labels = []
            xs = []
            for i, mask in enumerate(masks.squeeze(0)):
                label = label_fn(i, mask.cpu().detach().numpy())
                if label == "ignored":
                    continue
                labels.append(label)
                xs.append(slots[i].detach().cpu().numpy())
            if len(xs) != 0:
                self.PCA_dataset.append((np.array(xs), np.array(labels)))

    def split_PCA_dataset(self):
        x, y = zip(*self.PCA_dataset)
        return np.concatenate(x, axis=0), np.concatenate(y, axis=0)

    def learn_PCA(self):
        # assert PCA_dataset not empty and either list or tensor
        if self.PCA_dataset == []:
            raise Exception(
                "No PCA dataset yet. Call add_to_PCA_dataset() first")
        if not (type(self.PCA_dataset) == type([]) or type(self.PCA_dataset) == type(np.zeros(1))):
            raise Exception(
                f"PCA_dataset has wrong type {type(self.PCA_dataset)} (should be list or numpy.array). Call clear_PCA_dataset()")

        x, y = self.split_PCA_dataset()

        self.pca = self.PCA()
        self.pca.fit(x)

        try:
            self.principal_slot_components = self.pca.components_
        except:
            self.principal_slot_components = self.pca.eigenvectors_

    def PCA_dataset_2D(self, d=2, return_pca=False):
        """Perform PCA on the collected pca dataset
        if return_pca: returns, transformed x, y and the pca model
        else: return only transformed x, y
        """

        # assert PCA_dataset not empty and either list or tensor
        if self.PCA_dataset == []:
            raise Exception(
                "No PCA dataset yet. Call add_to_PCA_dataset() first")
        if not (type(self.PCA_dataset) == type([]) or type(self.PCA_dataset) == type(np.zeros(1))):
            raise Exception(
                f"PCA_dataset has wrong type {type(self.PCA_dataset)} (should be list or numpy.array). Call clear_PCA_dataset()")

        x, y = self.split_PCA_dataset()

        pca = self.PCA(n_components=d)
        if return_pca:
            return pca.fit_transform(x), y, pca
        return pca.fit_transform(x), y

    def modify(self, x, offset, modslot=0):
        if type(self.principal_slot_components) != type(np.zeros(1)):
            raise Exception(
                "principal_slot_components is not set. Call learn_PCA()")

        assert offset.shape == (self.principal_slot_components.shape[0], ) or \
            offset.shape == (self.principal_slot_components.shape[0], 1)
        # x has shape [batch_size, channels, width, height] = [batch_size, 3, 224, 224]
        features, x = self.encode(x)
        # features has shape [batch_size, num_patches, d_feat] = [batch_size, 784, 768]
        # x has shape [batch_size, num_patches, slot_dim]

        slots, masks = self.slot_attention(x)
        # slots has shape [batch_size, num_slots, slot_dim]

        slots[:, modslot, :] = slots[:, modslot, :] + \
            torch.tensor(self.principal_slot_components.T @
                         offset).reshape(1, -1).to(device)

        recon_combined, recons, masks = self.decode_fn(slots, features, masks)
        # recons has shape [batch_size, num_slots, num_patches, d_feat]
        # masks has shape [batch_size, num_slots, num_patches, 1]
        # recon_combined has shape [batch_size, num_patches, d_feat]

        return features, recon_combined, recons, masks, slots


# class SlotEmbedder(nn.Module):
#     def __init__(self, c_dim, device, use_c=True):
#         super().__init__()
#         encoder_name = "vit_base_patch8_224_dino"
#         slot_dim = 256
#         n_slots = 7
#         self.dinosaur = DINOSAUR(n_slots, 3, slot_dim, 2048, encoder_name=encoder_name,
#                                  use_transformer=True).to(device)
#         self.dinosaur.load_state_dict(torch.load(
#             f'./pg_modules/model_vit_base_patch8_224_dino_dataset.COCO_transformerTrue.ckpt')['model_state_dict'])
#         for param in self.dinosaur.parameters():
#             param.requires_grad = False
#         self.slots_to_c_shape = nn.Linear(slot_dim, c_dim)
#         self.combine_slots_and_c = nn.Linear(n_slots+(1 if use_c else 0), 1)
#         self.resize = Resize((224, 224))

#     def forward(self, c, img):
#         with torch.no_grad():
#             img = self.resize(img)
#             _, x = self.dinosaur.encode(img)
#             slots, _ = self.dinosaur.slot_attention(x)
#             slots = slots.detach()

#         slots = self.slots_to_c_shape(slots)
#         b, n, d = slots.shape

#         if c.shape[-1] == 0:
#             slots_and_c = slots.permute(0, 2, 1)
#         else:
#             slots_and_c = torch.cat(
#                 [slots, c.reshape(b, 1, d)], dim=1).permute(0, 2, 1)

#         return self.combine_slots_and_c(slots_and_c).squeeze(-1)
