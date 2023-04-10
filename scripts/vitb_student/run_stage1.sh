gpu_id=$1
data=${2:-0}
path=${3:-"../datasets/"}
lmd=${4:-0.5}

backbone="ViT-B/16"
dataset_list=("OfficeHome" "PACS" "VLCS" "TerraIncognita" "DomainNet")
dataset=${dataset_list[$data]}
echo "CLIP backbone: "$backbone
echo "Dataset: "$dataset

name="vitb-stage1-swad-"$backbone
CUDA_VISIBLE_DEVICES=$gpu_id python train_all.py $name \
    --clip_backbone $backbone \
    --backbone "vit-base" \
    --lmd $lmd \
    --data_dir $path \
    --algorithm DFC_STAGE1 \
    --dataset $dataset \
    --model_save 1000 \
    --swad True
