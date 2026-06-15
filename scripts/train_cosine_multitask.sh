export PATH=/usr/local/nvidia/bin:$PATH

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
#export NCCL_IB_HCA=mlx5_2,mlx5_5  # A100
export NCCL_IB_HCA=$(ibv_devices |tail -1 |awk '{print $1}') # V100
export NCCL_DEBUG=info
export OMP_NUM_THREADS=4
# ulimit -l 131072
export JOB_NAME=$(cat /etc/hostname | cut -d '-' -f 1,2,3)
#export MASTER_FILE=${MASTER_FILE_ROOT:-/tmp}/master_ip.${JOB_NAME}

# trap finish EXIT INT TERM

#export MASTER_ADDR=$(cat ${MASTER_FILE})
echo "master_ip: $MASTER_ADDR"
DIST_URL="tcp://$MASTER_ADDR:60900"


lr=1e-4
bs=2
uf=1
gpus=8
ep=10

dino_size=vit_large
dino_path=models/dinov2_vitl14_pretrain.pth
sam_size=vit_l
sam_path=models/sam_vit_l_0b3195.pth
clip_path=models/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin

SUB_DIR=cosine_lr${lr}_ep${ep}
echo ${SUB_DIR}

python tools/train.py \
    --num-gpus ${gpus} \
    --num-machines ${WORLD_SIZE} \
    --machine-rank ${RANK} \
    --dist-url ${DIST_URL} \
    --batch_size ${bs} \
    --epochs ${ep} \
    --update_freq ${uf} \
    --dinov2-size ${dino_size} \
    --dinov2-weights ${dino_path} \
    --sam-size ${sam_size} \
    --sam-weights ${sam_path} \
    --clip-weights ${clip_path} \
    --save_ckpt_freq 2 \
    --dataset "refer_seg" \
    --sample_rate "1" \
    --pano_seg_data "coco" \
    --pano_sample_rate "1" \
    --ins_seg_data "paco||o365" \
    --ins_sample_rate "2,3" \
    --refer_seg_data "refclef||refcoco||refcoco+||refcocog" \
    --refer_sample_rate "1,1,1,1"\
    --multimodal_choice "visual||text||visual_text" \
    --multimodal_rate "1,1,1" \
    --use_all_classes \
    --transformer_num_queries 200 \
    --transformer_fusion_layer_depth 1 \
    --crop_ratio 0.5 \
    --load_dir "" \
    --output_dir ./outputs/train/${SUB_DIR} \
    --log_dir ./outputs/train/${SUB_DIR} \
    --num_workers 2 \
    --auto_resume \
    --lr ${lr}
