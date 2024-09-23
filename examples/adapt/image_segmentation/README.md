# SegNeXt on cityscapes dataset using SSF 

## Overview 

This guide provides a walkthrough of applying SegNeXt for instance segmentation on the cityscapes dataset using SSF (Scaling and Shifting the deep Features). SSF enables parameter efficient fine-tuning by proposing that performace simillar to full fine-tuning can be achieved by only scaling and shifting the features of a deep neural network. 

## Table on Contents
 - [Introduction](#introduction)
 - [Requirements](#requirements)
 - [Installation](#installation)
 - [Dataset](#dataset)
 - [Configuration](#configuration)
 - [Adapting the model](#adapting-the-model)
 - [Conclusion](#conclusion)
 

## Introduction

In this example we will be finetuning SegNeXt large model on a city scapes dataset uisng a PEFT technique called SSF (Scale and Shift deep Features). SSF uses two linear layers to learn the scale and shift factors for deep features and hence uses very few trainable parameters. 

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

For this experiment we will be using the [CityScapes](https://www.kaggle.com/datasets/ipythonx/stanford-background-dataset) dataset. The original folder structure of the dataset loks like :  
```bash
iccv09Data
└── images
|   └── image1.jpg
|   └── image2.jpg
|   └── image3.jpg
    ...
└── seg_maps
|   └── image1.layers.txt
|   └── image1.regions.txt
|   └── image1.surfaces.txt
└── ann_file.txt
└── horizons.txt
```

The text files contain the segmentation maps in digits. In this dataset we will only be using the "regions" as the segmentation maps. These are the text files which are named as "image1.regions.txt". These text files are converted to ".png" images.  
The new data folder structure is formatted in the folowing way: 

```bash
iccv09Data
└── images
|   └── image1.jpg
|   └── image2.jpg
|   └── image3.jpg
    ...
└── seg_maps
    └── image1.png
    └── image2.png
    └── image3.png
    ...
└──train_ann.txt (text file containing the names of training images without the extension)
└──val_ann.txt (text file containing the names of validation images without the extension)
```

## Configuration

The following YAML file is used for setting up the experiment : 

```yaml

#DATASET
JOB_SERVICE: Adapt
CUSTOM_DATASET_PATH : "/custom_data/iccv09Data" 


MODEL : 'segnext' 
MODEL_PATH :  'segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512'  # config of model in case of mmseg
CACHE_BOOL : False
LOCAL_MODEL_PATH: #This is empty becasue we are using the pretrained model from the internet. 


SEED : 56
DO_TRAIN : True
DO_EVAL : False
NUM_WORKERS : 4
BATCH_SIZE : 1
EPOCHS : 1
STEPS : 
OPTIMIZER : 'AdamW' 
LR : 5e-4
SCHEDULER_TYPE : 'MultiStepLR'
WEIGHT_DECAY : 0.0
BETA1 : 0.9
BETA2 : 0.999
ADAM_EPS : 1e-8 
INTERVAL : 'steps'
INTERVAL_STEPS : 4      #For interval based training loops in mmlabs 
NO_OF_CHECKPOINTS : 5
FP16 : False
RESUME_FROM_CHECKPOINT : False 
GRADIENT_ACCUMULATION_STEPS : 3
GRADIENT_CHECKPOINTING : False
BEGIN: 0
END: 50

#MMSEG SPECIFIC ARGUMENTS
amp :  False
resume  : #False # ['auto']
auto_scale_lr :  False
cfg_options  : None
launcher :  none
dest_root : './.mmseg_cache'   #This saves the checkpoints and .py files for the model configs and mmseg logs   
train_ann_file : "train_ann_file.txt" 
val_ann_file : "val_ann_file.txt" 
work_dir : ./results/mmseg  # same as output dir
checkpoint_interval : 45
train_img_file: images  #For MMSEG - Folder containing training images
train_seg_file: seg_maps  #For MMSEG - Folder containing training segmentation maps
val_img_file: images   #For MMSEG - Folder containing validation images
val_seg_file: seg_maps     #For MMSEG - Folder containing validation segmentation maps
num_classes: 8
class_list: ['sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj'] #List containing all class names
palette: [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]   #List of lists contaning class colors - [[r,g,b],[r,g,b],[r,g,b]]


LAST_LAYER_TUNING : True
FULL_FINE_TUNING : False

PEFT_METHOD : 'SSF'
peft_type : 'SSF'


Library :  'MMSEG'
cuda_id : '0'
OUTPUT_DIR : "/user_data/jobs/Adapt/IMGSEG"
OVERWRITE_OUTPUT_DIR : False
LOGGING_PATH : "/user_data/logs/Adapt/IMGSEG" 
MERGE_ADAPTER: False
TASK : 'image_segmentation'
auto_select_modules: True
SAVE_METHOD : 'state_dict'

```

## Adapting the model
With the yaml file configured, the adaptation process is initiated with the following command : 

```bash 
nyun run examples/adapt/image_segmentation/config.yaml
```

Once the job starts, you will find the following directory structure in the `user_data` folder:

```bash
user_data/
├── jobs
│   └── Adapt
│       └── IMGSEG
├── logs
    └── Adapt
        └── IMGSEG
            └── log.log

```
The output model will be stored in `user_data/jobs/Adapt/IMGSEG/` directory and the final directory structure will be:

```bash
user_data/
├── jobs
│   └── Adapt
│       └── IMGSEG
│           └── merged_model_state_dict.pth
├── logs
    └── Adapt
        └── IMGSEG
            └── log.log

```

## Conclusion 

This guide has walked you through the process of adapting the SegNext model using SSF for instance segmentation of image regions on the cityscapes dataset. By employing SSF, we efficiently fine-tuned the model with a reduced memory footprint. The configuration and setup steps were outlined, ensuring that even complex tasks like distributed training and low-rank adaptation are manageable. The final trained model and logs are organized in a clear directory structure, making it easy to retrieve and analyze results.

---

*Author: [Panigrahi, Abhranta](https://www.linkedin.com/in/abhranta-panigrahi-626a23191/)*