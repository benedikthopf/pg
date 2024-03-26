from pg_modules import slot_attention
from pg_modules import dinosaur
import torch
from torch import nn
import torchvision.transforms
import timm


class SlotRecreater(nn.Module):
    def __init__(self, use_masks=False, oc_type="SA", feature_extractor="resnet18"):
        super().__init__()

        if oc_type == "SA":
            self.slot_dim = 64
            self.sa_resolution = (128, 128)
            self.sa = slot_attention.SlotAttentionAutoEncoder(
                self.sa_resolution, 7, 3, self.slot_dim)
            self.sa.load_state_dict(torch.load(
                f'../models/slotattention_clever.ckpt')['model_state_dict'])
        else:
            encoder_name = "vit_base_patch8_224_dino"
            self.slot_dim = 256
            n_slots = 7
            self.sa_resolution = (224, 224)
            self.sa = dinosaur.DINOSAUR(n_slots, 3, self.slot_dim, 2048, encoder_name=encoder_name,
                                        use_transformer=True)
            # self.sa.load_state_dict(torch.load(
            #     f'../models/model_vit_base_patch8_224_dino_dataset.COCO_transformerTrue.ckpt')['model_state_dict'])
            self.sa.load_state_dict(torch.load(
                f'../models/dinosaur_bedroom.ckpt')['model_state_dict'])

            # try:
            #     del self.sa.decoder
            # except:
            #     try:
            #         del self.sa.tf_dec
            #     except:
            #         raise RuntimeError("Model has no decoder")
        self.sa.requires_grad_(False)

        self.feature_extractor = timm.create_model(
            feature_extractor, num_classes=0)
        _, d = self.feature_extractor(torch.empty(1, 3, 256, 256)).shape
        self.features_to_slot_dim = nn.Linear(d, self.slot_dim)
        self.feature_resolution = (256, 256)

        self.feature_resize = torchvision.transforms.Resize(
            self.feature_resolution)
        self.sa_resize = torchvision.transforms.Resize(self.sa_resolution)

        self.use_masks = use_masks
        self.random_crop = torchvision.transforms.RandomCrop(
            self.feature_resolution, self.feature_resolution[0] // 3)

        if use_masks:
            cnn = []
            cnn.append(nn.Conv2d(1, 32, 1, padding="same"))
            cnn.append(nn.LeakyReLU())
            r = self.feature_resolution[0]
            while r >= 8:
                cnn.append(nn.Conv2d(32, 32, 3, dilation=2, padding="same"))
                cnn.append(nn.LeakyReLU())
                cnn.append(nn.MaxPool2d(2, 2))
                r = r // 2

            self.masks_cnn = nn.Sequential(*cnn)

            self.masks_mlp = nn.Sequential(
                nn.Linear(r*r*32, 256),
                nn.LeakyReLU(),
                nn.Linear(256, self.slot_dim),
                nn.LeakyReLU()
            )

            self.combine = nn.Linear(2*self.slot_dim, self.slot_dim)

    def put_in_center(self, imgs):
        *other_dims, c, h, w = imgs.shape
        imgs = imgs.reshape(-1, c, h, w)

        h_max = imgs.max(dim=1).values.max(dim=-1).values
        w_max = imgs.max(dim=1).values.max(dim=-2).values

        h_max_ceil = torch.where(h_max > 0.1, 1.0, 0.0)
        w_max_ceil = torch.where(w_max > 0.1, 1.0, 0.0)

        ys1 = h_max_ceil.argmax(dim=-1)
        ys2 = h - 1 - torch.flip(h_max_ceil, [-1,]).argmax(dim=-1)
        xs1 = w_max_ceil.argmax(dim=-1)
        xs2 = w - 1 - torch.flip(w_max_ceil, [-1,]).argmax(dim=-1)

        dxs = (xs2 - xs1)
        dys = (ys2 - ys1)

        out = torch.zeros_like(imgs)
        for i, (x1, y1, dx, dy, img) in enumerate(zip(xs1, ys1, dxs, dys, imgs)):
            # x_off = torch.randint(0, (w-dx).item(), (1, )).to(dy.device)
            # y_off = torch.randint(0, (w-dy).item(), (1, )).to(dy.device)
            x_off = w//2 - dx.item()//2
            y_off = h//2 - dy.item()//2
            cutout = img[:, y1:y1+dy, x1:x1+dx]
            # if not ((dx == w) and (dy == h)):
            out[i, :, y_off:y_off+dy, x_off:x_off+dx] = cutout

        out = out.reshape(*other_dims, c, w, h)
        return out

    def boxify_masks(self, masks, threshold=0.5):
        b, n, h, w = masks.shape

        h_max = masks.max(dim=-1).values
        w_max = masks.max(dim=-2).values

        h_max_ceil = torch.where(h_max > threshold, 1.0, 0.0)
        w_max_ceil = torch.where(w_max > threshold, 1.0, 0.0)

        # y1 = h_max_ceil.argmax(dim=-1)
        # y2 = h - 1 - torch.flip(h_max_ceil, [-1,]).argmax(dim=-1)
        # x1 = w_max_ceil.argmax(dim=-1)
        # x2 = w - 1 - torch.flip(w_max_ceil, [-1,]).argmax(dim=-1)

        # dss = torch.stack([(y2 - y1), (x2 - x1)], dim=-1).max(dim=-1).values

        # out = torch.zeros_like(masks)
        # for i, (hs, ws, ds) in enumerate(zip(y1, x1, dss)):
        #     for j, (h, w, d) in enumerate(zip(hs, ws, ds)):
        #         out[i, j, h:h+d, w:w+d] = 1

        h_max_ceil = h_max_ceil.reshape(b, n, h, 1)
        w_max_ceil = w_max_ceil.reshape(b, n, 1, w)

        out = h_max_ceil * w_max_ceil

        return out

    def forward(self, x, center=True):
        with torch.no_grad():
            *_, masks, slots = self.sa(self.sa_resize(x))
            if len(masks.shape) == 4:  # DINOSAUR
                b, s, n, _ = masks.shape
                wh = int(n**0.5)
                masks = masks.reshape(b, s, wh, wh, 1)
        # slots has shape [batch_size, n_slots, slot_dim]

        b, s, w, h, _ = masks.shape
        masks = masks.reshape(b*s, 1, w, h)

        masks = self.feature_resize(masks)
        x = self.feature_resize(x)

        w, h = self.feature_resolution
        masks = masks.reshape(b, s, 1, w, h)
        b, c, w, h = x.shape
        x = x.reshape(b, 1, c, w, h)

        masks[masks < 0.1] = 0

        masked_images = x * masks

        masked_images = masked_images.reshape(b*s, c, w, h)
        if center:
            masked_images = self.put_in_center(masked_images)

        ti_slots = self.feature_extractor(masked_images)

        ti_slots = ti_slots.reshape(b, s, -1)

        ti_slots = self.features_to_slot_dim(ti_slots)

        if self.use_masks:
            b, n, _, w, h = masks.shape
            masks = masks.reshape(b*n, 1, w, h)
            cnn_features = self.masks_cnn(masks)
            _, c, w, h = cnn_features.shape
            cnn_features = cnn_features.reshape(b*n, c*w*h)

            pos = self.masks_mlp(cnn_features)
            pos = pos.reshape(b, n, -1)

            ti_slots = self.combine(torch.cat([pos, ti_slots], dim=-1))

        return slots, ti_slots, {"masked_images": masked_images}
