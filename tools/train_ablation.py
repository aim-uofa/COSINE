import argparse
import os
import shutil
import sys
sys.path.append("./")
import datetime
import time
from functools import partial
import json
import copy
from typing import Any, Dict, List, Set
from pathlib import Path

import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import math

from cosine.data.dataset_new import build_dataset
from cosine.model.model_ablation import build_model
from cosine.utils.utils import Print
from cosine.utils import utils
from cosine.utils.launch import launch
import cosine.utils.comm as comm


def get_args():
    parser = argparse.ArgumentParser('COSINE Model Training')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--steps_per_epoch", default=1000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--output_dir', default='outputs/debug',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='outputs/debug',
                        help='path where to tensorboard log')
    parser.add_argument('--load_dir', default='',
                        help='path where to load parameters of stage 1')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')

    parser.add_argument("--lr", default=0.00001, type=float)
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9, 0.999, use opt default)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--data_root', default="datasets", type=str)
    parser.add_argument('--dataset', default="ins_seg||refer_seg", type=str)
    # parser.add_argument('--sample_rate', default="7,3", type=str)
    parser.add_argument('--ins_seg_data', default="coco||paco||o365", type=str)
    parser.add_argument('--ins_sample_rate', default="2,2,3", type=str)
    parser.add_argument('--refer_seg_data', default="refclef||refcoco||refcoco+||refcocog", type=str)
    parser.add_argument('--refer_sample_rate', default="1,1,1,1", type=str)
    parser.add_argument('--multimodal_choice', default="visual||text||visual_text||refer", type=str) # 'visual', 'text', 'visual_text'
    parser.add_argument('--multimodal_weight', default="1,1,1,1", type=str)
    parser.add_argument('--use_all_classes', action='store_true')
    # parser.set_defaults(use_all_classes=True)

    parser.add_argument('--random_flip', default="horizontal", type=str)
    parser.add_argument('--min_scale', default=0.1, type=float)
    parser.add_argument('--max_scale', default=2.0, type=float)
    parser.add_argument('--min_size', default=(560, 588, 616, 644, 672, 700), type=tuple)
    parser.add_argument('--max_size', default=896, type=int)
    parser.add_argument('--image_size', default=896, type=int)
    parser.add_argument('--sam_image_size', default=1024, type=int)
    parser.add_argument('--clip_image_size', default=1024, type=int)
    parser.add_argument('--crop_ratio', default=0.5, type=float)

    parser.add_argument('--feat_chans', default=256, type=int)
    parser.add_argument('--image_enc_use_fc', action="store_true")

    # parser.add_argument('--pt_model', type=str, default="dinov2")
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sam-size', type=str, default="vit_l")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_l_0b3195.pth")
    parser.add_argument('--clip-weights', type=str, default="models/CLIP-convnext_large_d_320.laion2B-s29B-b121K-ft-soup/open_clip_pytorch_model.bin")
    parser.add_argument('--img_feats_merging_type', type=str, default="simple_merging")


    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--transformer_nheads', default=8, type=int)
    parser.add_argument('--transformer_mlp_dim', default=2048, type=int)
    parser.add_argument('--transformer_mask_dim', default=256, type=int)
    parser.add_argument('--transformer_fusion_layer_depth', default=1, type=int)
    parser.add_argument('--transformer_num_queries', default=20, type=int)
    parser.add_argument("--transformer_pre_norm", action="store_true", default=True)

    parser.add_argument('--class_weight', default=2.0, type=float)
    parser.add_argument('--mask_weight', default=5.0, type=float)
    parser.add_argument('--dice_weight', default=5.0, type=float)
    parser.add_argument('--no_object_weight', default=0.1, type=float)
    parser.add_argument('--train_num_points', default=12544, type=int)
    parser.add_argument('--oversample_ratio', default=3.0, type=float)
    parser.add_argument('--importance_sample_ratio', default=0.75, type=float)
    parser.add_argument("--deep_supervision", action="store_true", default=True)
    # evaluation
    parser.add_argument('--score_threshold', default=0.8, type=float)

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--num-gpus", type=int, default=2, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )

    parser.add_argument('--enable_deepspeed', action='store_true', default=True)
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            Print("Please install deepspeed")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """

    new_batch = {}
    for b in batch:
        for k in b:
            if k not in new_batch:
                new_batch[k] = []
            new_batch[k].append(b[k])

    return new_batch

def main(args, ds_init):

    args.gpu = comm.get_local_rank()
    args.world_size = utils.get_world_size()
    args.local_rank = comm.get_local_rank()

    os.environ['LOCAL_RANK'] = str(args.gpu)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ["RANK"] = str(args.rank)

    if ds_init is not None:
        utils.create_ds_config(args)

    Print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    Print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        Print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=trivial_batch_collator
    )

    model = build_model(args)
    model.to(device)

    if args.load_dir:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(args.load_dir)
        msg = model.load_state_dict(state_dict, strict=False)
        Print(f"Loading model parameters msg: {msg}")

    model_without_ddp = model
    Print("Model = %s" % str(model_without_ddp))

    learnable_param_list = [_ for _, p in model.named_parameters() if p.requires_grad]
    Print(f"learnable params: {learnable_param_list}")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Print('number of params (M): %.2f' % (n_parameters / 1.e6))

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    assert num_training_steps_per_epoch == args.steps_per_epoch
    Print("LR = %.8f" % args.lr)
    Print("Batch size = %d" % total_batch_size)
    Print("Update frequent = %d" % args.update_freq)
    Print("Number of training examples = %d" % len(dataset_train))
    Print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    def get_default_optimizer_params(args, model):

        weight_decay_norm = 0.0 # args.weight_decay_norm
        weight_decay_embed = 0.0 # args.weight_decay_embed

        defaults = {}
        defaults["lr"] = args.lr
        defaults["weight_decay"] = args.weight_decay

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * 0.1 # BACKBONE_MULTIPLIER
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    Print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
        return params


    if args.distributed:
        torch.distributed.barrier()
    if args.enable_deepspeed:

        optimizer_params = get_default_optimizer_params(args, model)
        model, optimizer, _, scheduler = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )
    else:
        raise NotImplementedError

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer
    )

    Print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            scheduler,
            device,
            epoch,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            args=args,
        )
        torch.cuda.empty_cache()

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    Print('Training time {}'.format(total_time_str))

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale, optimizer._global_grad_norm

def train_one_epoch(
        model: torch.nn.Module,
        data_loader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        epoch: int,
        loss_scaler=None,
        log_writer=None,
        start_steps=None,
        num_training_steps_per_epoch=None,
        update_freq=None,
        args=None
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    multimodal_choice = [x for x in args.multimodal_choice.split("||")]
    multimodal_weight = [float(x) for x in args.multimodal_weight.split(",")]
    assert len(multimodal_choice) == len(multimodal_weight)

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, multi_inputs in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue

        loss_dict = {}
        for prompt, prompt_weight in zip(multimodal_choice, multimodal_weight):
            inputs = multi_inputs[prompt]

            if loss_scaler is None:
                if args.precision == "fp16":
                    for input in inputs:
                        input['target']['image'] = input['target']['image'].half()
                        input['target']['sam_image'] = input['target']['sam_image'].half()
                        input['target']['clip_image'] = input['target']['clip_image'].half()
                        if input['visual_prompt'] is not None:
                            input['visual_prompt']['image'] = input['visual_prompt']['image'].half()
                            input['visual_prompt']['sam_image'] = input['visual_prompt']['sam_image'].half()
                            input['visual_prompt']['clip_image'] = input['visual_prompt']['clip_image'].half()
                elif args.precision == "bf16":
                    for input in inputs:
                        input['target']['image'] = input['target']['image'].bfloat16()
                        input['target']['sam_image'] = input['target']['sam_image'].bfloat16()
                        input['target']['clip_image'] = input['target']['clip_image'].bfloat16()
                        if input['visual_prompt'] is not None:
                            input['visual_prompt']['image'] = input['visual_prompt']['image'].bfloat16()
                            input['visual_prompt']['sam_image'] = input['visual_prompt']['sam_image'].bfloat16()
                            input['visual_prompt']['clip_image'] = input['visual_prompt']['clip_image'].bfloat16()
                else:
                    for input in inputs:
                        input['target']['image'] = input['target']['image'].float()
                        input['target']['sam_image'] = input['target']['sam_image'].float()
                        input['target']['clip_image'] = input['target']['clip_image'].float()
                        if input['visual_prompt'] is not None:
                            input['visual_prompt']['image'] = input['visual_prompt']['image'].float()
                            input['visual_prompt']['sam_image'] = input['visual_prompt']['sam_image'].float()
                            input['visual_prompt']['clip_image'] = input['visual_prompt']['clip_image'].float()
            else:
                raise NotImplementedError

            loss_dict_ = model(inputs)
            for k in loss_dict_:
                if k not in loss_dict:
                    loss_dict[k] = loss_dict_[k] * prompt_weight
                else:
                    loss_dict[k] = loss_dict[k] + loss_dict_[k] * prompt_weight

        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / len(multimodal_choice)

        losses = sum(loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            Print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            losses /= update_freq
            model.backward(losses)
            model.step()

            grad_norm = None
            loss_scale_value, grad_norm = get_loss_scale_for_deepspeed(model)
        else:
            raise NotImplementedError

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**loss_dict)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(**loss_dict, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    Print(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    launch(
        main,
        opts.num_gpus,
        num_machines=opts.num_machines,
        machine_rank=opts.machine_rank,
        dist_url=opts.dist_url,
        args=(opts, ds_init),
    )