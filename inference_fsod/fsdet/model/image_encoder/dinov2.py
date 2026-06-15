import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class DINOv2EncoderViT(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        out_chans: int = 256,
        use_fc: bool = False
    ):
        super().__init__()
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        if not use_fc:
            self.neck = nn.Sequential(
                nn.Conv2d(
                    encoder.embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
                nn.Conv2d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
            )
        else:
            self.neck = nn.Conv2d(
                    encoder.embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                )

    def get_enc_embs(self, pixel_values: torch.FloatTensor):
        b, _, h, w = pixel_values.shape
        h, w = h // self.encoder.patch_size, w // self.encoder.patch_size
        image_embeddings = self.encoder.forward_features(pixel_values)["x_prenorm"][:, 1:]
        image_embeddings = image_embeddings.permute(0, 2, 1).contiguous().reshape(b, -1, h, w) # b, c, h, w

        return image_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.get_enc_embs(x)
        x = self.neck(x)
        return x
