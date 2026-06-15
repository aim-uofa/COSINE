import os
import torch

from tqdm import tqdm

from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader

import sys
sys.path.append('./')

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup
from fsdet.data.dataset_mapper_fsl import DatasetMapper
from fsdet.cosine_config import add_cosine_config

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

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)

    trainloader = build_detection_test_loader(cfg, cfg.DATASETS.TRAIN[0], mapper=DatasetMapper(cfg, False), batch_size=cfg.SOLVER.IMS_PER_BATCH)

    model = build_model(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loading model parameters msg: {msg}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(trainloader)):
            new_batch = [b for b in batch if len(b['instances']) > 0]
            if len(new_batch)==0:
                continue
            model(new_batch)
        id_labels, id_queries, seg_labels, \
        seg_queries = model.integrate_queries()

    os.makedirs(
        os.path.join(cfg.OUTPUT_DIR), exist_ok=True
    )

    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth')
    state_dict = model.state_dict()
    new_state_dict = {
        'id_labels': state_dict['id_labels'],
        'id_queries': state_dict['id_queries'],
        'seg_labels': state_dict['seg_labels'],
        'seg_queries': state_dict['seg_queries'],
    }

    to_save = {
        'model': new_state_dict,
    }
    torch.save(to_save, checkpoint_path)
