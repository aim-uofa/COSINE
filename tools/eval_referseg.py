import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import tqdm

sys.path.append("./")

from inference_referseg.data.dataset import ValDataset
from inference_referseg.model.model import build_model
from inference_referseg.utils.launch import launch
import inference_referseg.utils.comm as comm
from inference_referseg.utils.utils import AverageMeter, Summary, intersectionAndUnionGPU, dict_to_cuda

def get_args_parser():
    # Arguments parsing
    parser = argparse.ArgumentParser(description='COCOSINE PyTorch Implementation for Refereing Segmentation')

    # Dataset parameters
    parser.add_argument("--val_dataset", default="refcoco|unc|val", type=str)
    parser.add_argument("--dataset_dir", default="./datasets", type=str)

    parser.add_argument('--image_size', type=int, default=896)
    parser.add_argument('--sam_image_size', type=int, default=1024)
    parser.add_argument('--clip_image_size', type=int, default=1024)

    parser.add_argument('--output_dir', type=str, default='outputs/referseg/debug')

    # Model parameters
    parser.add_argument('--feat_chans', default=256, type=int)
    parser.add_argument('--image_enc_use_fc', action="store_true")
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--weights', type=str, default="models/cosine/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin")
    parser.add_argument('--sam-size', type=str, default="vit_b")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_b_01ec64.pth")
    parser.add_argument('--clip-weights', type=str, default="models/CLIP-convnext_large_d_320.laion2B-s29B-b121K-ft-soup/open_clip_pytorch_model.bin")

    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--transformer_nheads', default=8, type=int)
    parser.add_argument('--transformer_mlp_dim', default=2048, type=int)
    parser.add_argument('--transformer_mask_dim', default=256, type=int)
    parser.add_argument('--transformer_fusion_layer_depth', default=1, type=int)
    parser.add_argument('--transformer_num_queries', default=200, type=int)
    parser.add_argument("--transformer_pre_norm", action="store_true", default=True)
    parser.add_argument('--score_threshold', default=-1e9, type=float)

    # distributed training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
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

    return parser

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """

    return batch

def main(args):
    args.gpu = comm.get_local_rank()

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    if args.local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(args.output_dir)
    else:
        writer = None


    # dataset
    val_dataset = ValDataset(
        base_image_dir=args.dataset_dir,
        val_dataset=args.val_dataset,
        image_size=args.image_size,
        sam_image_size=args.sam_image_size,
        clip_image_size=args.clip_image_size
    )
    if args.dist_eval:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    assert args.batch_size == 1
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=trivial_batch_collator
    )

    # Model initialization
    model = build_model(args)
    print(f"model: {model}")
    state_dict = torch.load(args.weights, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"msg: {msg}")
    model.to(device)
    model.eval()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    validate(data_loader_val, model, 0, writer, args)



def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        for input in input_dict:
            input['target']['image'] = input['target']['image'].float()
            input['target']['sam_image'] = input['target']['sam_image'].float()
            input['target']['clip_image'] = input['target']['clip_image'].float()
            if input['visual_prompt'] is not None:
                input['visual_prompt']['image'] = input['visual_prompt']['image'].float()
                input['visual_prompt']['sam_image'] = input['visual_prompt']['sam_image'].float()
                input['visual_prompt']['clip_image'] = input['visual_prompt']['clip_image'].float()

        with torch.no_grad():
            output_dict = model_engine(input_dict)

        pred_masks = output_dict["id_seg"].pred_masks
        masks_list = input_dict[0]['target']["gt_masks"].int().to(pred_masks.device)
        output_list = (pred_masks > 0).int()
        assert len(output_list) == len(masks_list)

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    if args.dist_eval:
        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        res = "giou: {:.4f}, ciou: {:.4f}".format(giou, ciou)
        print(res)
        with open(os.path.join(args.output_dir, 'result.txt'), 'w') as file:
            file.write(res)

    return giou, ciou

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )