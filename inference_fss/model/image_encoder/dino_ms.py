import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Type

from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils

from detectron2.modeling import ShapeSpec
from detectron2.layers import Conv2d, get_norm

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

def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

class DINOv2EncoderViT(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        out_chans: int = 256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=None,
        norm="LN",
    ):
        super().__init__()

        self.patch_size = encoder.patch_size
        self.out_chans = out_chans
        self.encoder = encoder

        self.scale_factors = scale_factors

        strides = [int(16 / scale) for scale in scale_factors]
        _assert_strides_are_log2_contiguous(strides)

        dim = encoder.embed_dim

        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_chans,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_chans),
                    ),
                    Conv2d(
                        out_chans,
                        out_chans,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_chans),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

            if stage == 4:
                self.neck = layers

        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_chans for k in self._out_features}

    def get_enc_embs(self, pixel_values: torch.FloatTensor):
        b, _, h, w = pixel_values.shape
        h, w = h // self.encoder.patch_size, w // self.encoder.patch_size
        image_embeddings = self.encoder.forward_features(pixel_values)["x_prenorm"][:, 1:]
        image_embeddings = image_embeddings.permute(0, 2, 1).contiguous().reshape(b, -1, h, w) # b, c, h, w

        return image_embeddings

    def forward_neck(self, x):
        results = []
        for stage in self.stages:
            results.append(stage(x))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.get_enc_embs(x)
        results = []
        for stage in self.stages:
            results.append(stage(x))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in ["p2", "p3", "p4", "p5"]
        }


def build_dinov2(args):

    dinov2_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )

    dinov2 = vits.__dict__[args.dinov2_size](**dinov2_kwargs)

    dinov2_utils.load_pretrained_weights(dinov2, args.dinov2_weights, "teacher")
    dinov2.eval()
    for param in dinov2.parameters():
        param.requires_grad = False
    dino_encoder = DINOv2EncoderViT(dinov2, out_chans=args.feat_chans)
    return dino_encoder
