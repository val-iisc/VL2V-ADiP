# VL2V-ADiP: Vision-Language to Vision - Align, Distill, Predict

## Preparation

### Dependencies

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
