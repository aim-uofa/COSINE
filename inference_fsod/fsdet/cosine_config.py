# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_cosine_config(cfg):
    # Matcher FSOD inference config.
    cfg.MODEL.META_ARCHITECTURE = "COSINE"
    cfg.MODEL.DINO = CN(new_allowed=True)
    cfg.MODEL.DINO.WEIGHTS = ""
    cfg.MODEL.DINO.OUT_CHANS = 256
    cfg.MODEL.CLIP = CN(new_allowed=True)
    cfg.MODEL.CLIP.WEIGHTS = ""
    cfg.MODEL.COSINE = CN(new_allowed=True)
    cfg.MODEL.COSINE.sem_seg_postprocess_before_inference = True
    cfg.MODEL.COSINE.test_topk_per_image = 100
    cfg.MODEL.COSINE.score_threshold = 0.7
    cfg.MODEL.Transformer = CN(new_allowed=True)
    cfg.MODEL.PIXEL_DECODER = CN(new_allowed=True)

    # Input.
    cfg.INPUT.MIN_SIZE_TEST = 896
    cfg.INPUT.MAX_SIZE_TEST = 896
    cfg.INPUT.IMAGE_SIZE = 896
    cfg.INPUT.SAM_IMAGE_SIZE = 1024
    cfg.INPUT.CLIP_IMAGE_SIZE = 1024
