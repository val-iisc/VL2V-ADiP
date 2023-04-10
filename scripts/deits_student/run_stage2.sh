gpu_id=$1
data=${2:-0}
path=${3:-"../datasets/"}
lmd=${4:-0.5}

backbone="ViT-B/16"
dataset_list=("OfficeHome" "PACS" "VLCS" "TerraIncognita" "DomainNet")
dataset=${dataset_list[$data]}
name="deits-stage2-swad-"$backbone
echo "CLIP backbone: "$backbone
echo "Dataset: "$dataset

pretrain_pth="train_output/"$dataset"/deits-stage1-swad-"$backbone"/checkpoints/"
CUDA_VISIBLE_DEVICES=$gpu_id python train_all.py $name \
    --clip_backbone $backbone \
    --model_save 100 \
    --lmd $lmd \
    --data_dir $path \
    --backbone "deit-small" \
    --algorithm DFC_STAGE2 \
    --dataset $dataset \
    --pretrained $pretrain_pth \
    --swad True
