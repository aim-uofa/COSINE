"""
Detection Testing Script.

This scripts reads a given config file and runs the evaluation.
It is an entry point that is made to evaluate standard models in FsDet.

In order to let one script support evaluation of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import logging
import json
import os
import time

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import launch
from detectron2.modeling import build_model
from detectron2.data.build import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator as D2COCOEvaluator

import sys
sys.path.append('./')

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup
from fsdet.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    # PascalVOCDetectionEvaluator,
    verify_results,
)
from fsdet.evaluation.my_pascal_evaluation import PascalVOCDetectionEvaluator
from fsdet.data.dataset_mapper_fsl import DatasetMapper
from fsdet.cosine_config import add_cosine_config


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if 'VOCSegm' in dataset_name and evaluator_type == "coco":
            evaluator_list.append(
                D2COCOEvaluator(dataset_name, cfg, True, output_folder)
            )
        elif evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder)
            )
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`fsdet.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        if not cfg.MUTE_HEADER:
            logger = logging.getLogger(__name__)
            logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`fsdet.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))


class Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = Trainer.build_model(cfg)
        self.check_pointer = DetectionCheckpointer(
            self.model, save_dir=cfg.OUTPUT_DIR
        )
        state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"Loading model parameters msg: {msg}")

        self.best_res = None
        self.best_file = None
        self.all_res = {}

    def test(self, ckpt):
        model = torch.load(ckpt)['model']
        self.model.register_buffer("id_labels", model['id_labels'].to(self.model.device))
        self.model.register_buffer("id_queries", model['id_queries'].to(self.model.device))
        self.model.register_buffer("seg_labels", model['seg_labels'].to(self.model.device))
        self.model.register_buffer("seg_queries", model['seg_queries'].to(self.model.device))
        sorted_indices = torch.argsort(self.model.seg_labels)
        self.model.seg_labels = self.model.seg_labels[sorted_indices]
        self.model.seg_queries = self.model.seg_queries[sorted_indices]
        print("evaluating checkpoint {}".format(ckpt))
        res = Trainer.test(self.cfg, self.model)

        if comm.is_main_process():
            verify_results(self.cfg, res)
            print(res)
            if (self.best_res is None) or (
                self.best_res is not None
                and self.best_res["bbox"]["AP"] < res["bbox"]["AP"]
            ):
                self.best_res = res
                self.best_file = ckpt
            print("best results from checkpoint {}".format(self.best_file))
            print(self.best_res)
            self.all_res["best_file"] = self.best_file
            self.all_res["best_res"] = self.best_res
            self.all_res[ckpt] = res
            os.makedirs(
                os.path.join(self.cfg.OUTPUT_DIR, "inference"), exist_ok=True
            )
            with open(
                os.path.join(self.cfg.OUTPUT_DIR, "inference", "all_res.json"),
                "w",
            ) as fp:
                json.dump(self.all_res, fp)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_cosine_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    tester = Tester(cfg)
    if args.smoke_model_only:
        print("FSOD smoke-model-only succeeded.")
        return {}
    ckpt = os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth')
    tester.test(ckpt)
    return tester.best_res


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if args.eval_during_train or args.eval_all:
        args.dist_url = "tcp://127.0.0.1:{:05d}".format(
            np.random.choice(np.arange(0, 65534))
        )
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
