gpu_id=$1
data=${2:-0}
path=${3:-"../datasets/"}
lmd=${4:-0.5}
seed=${5:-0}

backbone="ViT-B/16"
dataset_list=("OfficeHome" "PACS" "VLCS" "TerraIncognita" "DomainNet")
dataset=${dataset_list[$data]}
echo "CLIP backbone: "$backbone
echo "Dataset: "$dataset

name="vitb-stage2-swad-"$backbone
pretrain_pth="train_output/"$dataset"/vitb-stage1-swad-"$backbone"/checkpoints/"
CUDA_VISIBLE_DEVICES=$gpu_id python train_all.py $name \
    --clip_backbone $backbone \
    --swad_fix \
    --lmd $lmd \
    --seed $seed \
    --model_save 100 \
    --data_dir $path \
    --backbone "vit-base" \
    --algorithm DFC_STAGE2 \
    --dataset $dataset \
    --pretrained $pretrain_pth \
    --swad True
