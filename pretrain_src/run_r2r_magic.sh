
name=r2r_MAGIC_pretrain
DATA_ROOT=../datasets/R2R/
NODE_RANK=0
NUM_GPUS=2

# train
CUDA_VISIBLE_DEVICES='1,2' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port 8887 \
    train_r2r_magic.py --world_size ${NUM_GPUS} \
    --name ${name} \
    --vlnbert cmt \
    --model_config config/r2r_magic_model_config.json \
    --config config/r2r_magic_pretrain.json \
    --root_dir $DATA_ROOT \
    --cuda_first_device 1