# Leveraging Vision-Language Models for Improving Domain Generalization in Image Classification

This repository contains the code for the paper [Leveraging Vision-Language Models for Improving Domain Generalization in Image Classification](https://arxiv.org/abs/2310.08255) by Sravanti Addepalli*, Ashish Ramayee Asokan*, Lakshay Sharma, and R. Venkatesh Babu.

TLDR: We propose to leverage the superior generalization ability of Vision-Language models to improve OOD generalization in image classification. We obtain SOTA results in both black box and white box settings of the Vision-Language model!

### Abstract

Vision-Language Models (VLMs) such as CLIP are trained on large amounts of image-text pairs, resulting in remarkable generalization across several data distributions. The prohibitively expensive training and data collection/curation costs of these models make them valuable Intellectual Property (IP) for organizations. This motivates a vendor-client paradigm, where a vendor trains a large-scale VLM and grants only input-output access to clients on a pay-per-query basis in a black-box setting. The client aims to minimize inference cost by distilling the VLM to a student model using the limited available task-specific data, and further deploying this student model in the downstream application. While naive distillation largely improves the In-Domain (ID) accuracy of the student, it fails to transfer the superior out-of-distribution (OOD) generalization of the VLM teacher using the limited available labeled images. To mitigate this, we propose Vision-Language to Vision-Align, Distill, Predict (VL2V-ADiP), which first aligns the vision and language modalities of the teacher model with the vision modality of a pre-trained student model, and further distills the aligned VLM embeddings to the student. This maximally retains the pre-trained features of the student, while also incorporating the rich representations of the VLM image encoder and the superior generalization of the text embeddings. The proposed approach achieves state-of-the-art results on the standard Domain Generalization benchmarks in a black-box teacher setting, and also when weights of the VLM are accessible.

## Code

### Installing dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

1. To run the first stage (Align)

```
bash scripts/<stud_model>_student/run_stage1.sh <gpu_id> <dataset_id> <dataset_path>
```

2. To run the second stage (Distill)

- Rename the stage 1 folder by removing the additional info as shown in the below example:
```
232316_11-22-29_vitb-stage1-swad-ViT-B --> vitb-stage1-swad-ViT-B
```
- Run the following command:
```
bash scripts/<stud_model>_student/run_stage2.sh <gpu_id> <dataset_id> <dataset_path>
```

Here, ```<stud_model>``` refers to the student architecture, which can be one of the following: rn50, vitb, vits, deits. ```<gpu_id>``` is the GPU ID of the machine. ```<dataset_id>``` is the dataset for the experiment indexed in the following order: (OfficeHome, PACS, VLCS, TerraIncognita, DomainNet). ```<dataset_path>``` is the parent folder of all the datasets given with the "/" included. (Eg: /my_folder/datasets/) 


## License

This project is released under the MIT license

This project include some code from [facebookresearch/DomainBed](https://github.com/facebookresearch/DomainBed) (MIT license) and
[khanrc/swad](https://github.com/khanrc/swad) (MIT license) and [kakaobrain/MIRO](https://github.com/kakaobrain/miro) (MIT license).
