# Finetuning RTMDet on face-det dataset using LoRA and DDP

## Overview

This guide provides a walkthrough of applying RTMDet for face detection on the face-det dataset using LoRA (Low-Rank Adaptation) with Distributed Data Parallel (DDP) across 2 GPUs. LoRA enables efficient fine-tuning by reducing the memory footprint, making it a powerful approach for high-performance face detection while maintaining scalability and resource efficiency.

## Table on Contents
 - [Introduction](#introduction)
 - [Requirements](#requirements)
 - [Installation](#installation)
 - [Dataset](#dataset)
 - [Configuration](#configuration)
 - [Adapting the model](#adapting-the-model)
 - [Conclusion](#conclusion)
 

## Introduction 

In this example, we'll be applying the LoRA technique to fine-tune the RTMDet model on the Face-Det dataset. The training process will be distributed across 2 GPUs using Distributed Data Parallel (DDP) to maximize efficiency. LoRA's ability to introduce low-rank adaptations allows for targeted model updates, significantly reducing memory and computational requirements while increasing (Out Of Distribution) OOD generalization performance. 

## Requirements

Before you begin, ensure that you have the following:
- A GPU-enabled environment with CUDA support.
- The Nyuntam repository cloned and set up as per the [Installation Guide](#installation).
- Docker

## Installation

### Step 1: Clone the Nyuntam Repository

Clone the repository and navigate to the `nyuntam` directory:
```bash
$ git clone https://github.com/nyunAI/nyuntam.git
$ cd nyuntam
```

### Step 2: Set Up the workspace
To setup the environment use the following command(s),

```bash
pip install git+https://github.com/nyunAI/nyunzero-cli.git
nyun init {WORKSPACE_PATH} -e adapt
```

## Dataset

For this experiment we will be using the [face-det](https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i) dataset from roboflow. The dataset is downloaded in coco format. The data folder structure is formatted in the folowing way: 

```bash
face-det
└── train
|   └── annotations_coco.json
|   └── image1.jpg
|   └── image2.jpg
    ...
└── validation
    └── annotations_coco.json
    └── image1.jpg
    └── image2.jpg

```

## Configuration

The following YAML file is used for setting up the experiment : 

```yaml
#DATASET_ARGS :
JOB_SERVICE: Adapt
TRAIN_DIR : 'train/'
VAL_DIR : 'validation/'
TEST_DIR : 
CUSTOM_DATASET_PATH : "/custom_data/face_det"

#MODEL_ARGS :
MODEL : 'rtmdet' 
MODEL_PATH :  'rtmdet_tiny_8xb32-300e_coco'  # config of model in case of mmdet
CACHE_BOOL : False
LOCAL_MODEL_PATH:  #This is empty becasue we are using the pretrained model from the internet. 

#TRAINING_ARGS :
SEED : 56
DO_TRAIN : True
DO_EVAL : True
NUM_WORKERS : 4
BATCH_SIZE : 1
EPOCHS : 1
OPTIMIZER : 'SGD' 
LR : 5e-3 
SCHEDULER_TYPE : 'CosineAnnealingLR'
WEIGHT_DECAY : 0.0
INTERVAL : 'steps'
INTERVAL_STEPS : 4
NO_OF_CHECKPOINTS : 5
FP16 : False
GRADIENT_ACCUMULATION_STEPS : 4
GRADIENT_CHECKPOINTING : False
BEGIN: 0
END: 50

#MMDET SPECIFIC ARGUMENTS :
amp :  False
resume  : False # ['auto']
auto_scale_lr :  False
cfg_options  : None
launcher :  pytorch
dest_root : './.mmdet_cache'       
train_ann_file : 'annotations.coco.json'     
val_ann_file : 'annotations.coco.json'        
work_dir : ./results/mmdet  # same as output dir
checkpoint_interval : 5

#FINE_TUNING_ARGS :
LAST_LAYER_TUNING : True
FULL_FINE_TUNING :  False

PEFT_METHOD : "LoRA"

#LoRA_CONFIG :
r : 32
alpha : 16
dropout : 0.1
peft_type : 'LoRA'
fan_in_fan_out : False
init_lora_weights : True  

TASK : 'object_detection'
Library :  'MMDET' 
cuda_id : '0,1'
OUTPUT_DIR : "/user_data/jobs/Aadpt/OBJDET"
LOGGING_PATH : "/user_data/logs/Adapt/OBJDET"
MERGE_ADAPTER: True
auto_select_modules: True
SAVE_METHOD : 'state_dict'

#DDP ARGS
DDP: True
num_nodes: 1
```
## Adapting the model
With the yaml file configured, the adaptation process is initiated with the following command : 

```bash 
nyun run examples/adapt/object_detection/config.yaml
```

Once the job starts, you will find the following directory structure in the `/user_data` folder:

```bash
/user_data/
├── jobs
│   └── Adapt
│       └── OBJDET
├── logs
    └── Adapt
        └── OBJDET
            └── log.log

```

LoRA decreases the total trainable parameters of the model by freezing the original model weights and updating LoRA adapters only. 
```log 
08/05/2024 23-04-20 - INFO - nyuntam_adapt.core.base_algorithm - trainable params: 1491594 || all params: 6009438 || trainable%: 24.82
```


This is a sample of the experiment logs  : 

```log
08/05/2024 23-04-34 - INFO - stdout - 08/05 23:04:34 - mmengine - INFO - Epoch(train) [1][ 150/2988]  base_lr: 5.0000e-03 lr: 5.0000e-03  eta: 0:03:39  time: 0.0673  data_time: 0.0008  memory: 220  loss: 2.0014  loss_cls: 0.9983  loss_bbox: 1.0031

08/05/2024 23-04-37 - INFO - stdout - 08/05 23:04:37 - mmengine - INFO - Epoch(train) [1][ 200/2988]  base_lr: 5.0000e-03 lr: 5.0000e-03  eta: 0:03:29  time: 0.0683  data_time: 0.0008  memory: 220  loss: 1.8670  loss_cls: 0.8558  loss_bbox: 1.0112

08/05/2024 23-04-40 - INFO - stdout - 08/05 23:04:40 - mmengine - INFO - Epoch(train) [1][ 250/2988]  base_lr: 5.0000e-03 lr: 5.0000e-03  eta: 0:03:22  time: 0.0695  data_time: 0.0008  memory: 220  loss: 1.7820  loss_cls: 0.8524  loss_bbox: 0.9297

```
The output model will be stored in `/user_data/jobs/Adapt/100/` directory and the final directory structure will be:

```bash
/user_data/
├── jobs
│   └── Adapt
│       └── OBJDET
│           └── merged_model_state_dict.pth
├── logs
    └── Adapt
        └── OBJDET
            └── log.log

```

## Conclusion 

This guide has walked you through the process of adapting the RTMDet model using LoRA for face detection on the Face-Det dataset, leveraging Distributed Data Parallel (DDP) across two GPUs. By employing LoRA, we efficiently fine-tuned the model with a reduced memory footprint, allowing for scalable and high-performance object detection. The configuration and setup steps were outlined, ensuring that even complex tasks like distributed training and low-rank adaptation are manageable. The final trained model and logs are organized in a clear directory structure, making it easy to retrieve and analyze results.

---

*Author: [Panigrahi, Abhranta](https://www.linkedin.com/in/abhranta-panigrahi-626a23191/)*







