# Finetuning T5 large with QLoRA on XSUM dataset

## Overview 

This guide provides a detailed walkthrough for finetuning T5 Large model on the xsum Dataset with QLoRA using nyuntam-adapt. QLoRA is a PEFT technique where the original weights are frozen to reduce the trainable parameters and and qre quantized to reduce the memory usage. 

## Table on Contents
 - [Introduction](#introduction)
 - [Requirements](#requirements)
 - [Installation](#installation)
 - [Dataset](#dataset)
 - [Configuration](#configuration)
 - [Adapting the model](#adapting-the-model)
 - [Conclusion](#conclusion)

## Introduction
In this example we will be finetuning a T5 large model for text summarization on the xsum dataset using QLoRA. QLoRA (Quantized LoRA) allows us to finetune a large model with a small memory requirement by freezing and quantizing the original model weights and only training the LoRA adapters. The adapters are then merged while saving the model. 

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

[XSum Dataset](https://huggingface.co/datasets/EdinburghNLP/xsum) is used for this example. The dataset is directly used from 🤗hugginface. 

Sample : 
| document                                                                 | summary                                                                 | id        |
|--------------------------------------------------------------------------|--------------------------------------------------------------------------|-----------|
| The full cost of damage in Newton Stewart, one of the areas worst affec… | Clean-up operations are continuing across the Scottish Borders and Dumf… | 35232142  |
| A fire alarm went off at the Holiday Inn in Hope Street at about 04:20 … | Two tourist buses have been destroyed by fire in a suspected arson atta… | 40143035  |
| Ferrari appeared in a position to challenge until the final laps, when … | Lewis Hamilton stormed to pole position at the Bahrain Grand Prix ahead… | 35951548  |
| John Edward Bates, formerly of Spalding, Lincolnshire, but now living i… | A former Lincolnshire Police officer carried out a series of sex attack… | 36266422  |
| Patients and staff were evacuated from Cerahpasa hospital on Wednesday … | An armed man who locked himself into a room at a psychiatric hospital i… | 38826984  |


## Configuration

The following YAML file is used for setting up the experiment : 

```yaml
JOB_SERVICE : Adapt
JOB_ID: SUMM
TASK : Seq2Seq_tasks
subtask : summarization
max_input_length : 512
max_target_length : 128
eval_metric : 'rouge' 
cuda_id : '0'
OUTPUT_DIR : "/user_data/jobs/Adapt/SUMM"
OVERWRITE_OUTPUT_DIR : False
LOGGING_PATH: "/user_data/logs/Adapt/SUMM" 
packing : True
dataset_text_field : 'text' 
max_seq_length : 512
flash_attention2 : false
blocksize : 128
SAVE_METHOD : 'state_dict'


# DATASET_ARGS :
DATASET : 'EdinburghNLP/xsum'
DATA_VERSION : '1.0'
MAX_TRAIN_SAMPLES : 1000
MAX_EVAL_SAMPLES : 1000
DATASET_CONFIG : {}
input_column : 'document'
target_column : 'summary'

# MODEL_ARGS :
MODEL : "t5"
MODEL_PATH :  'google-t5/t5-large'
MODEL_VERSION : '1.0'
CACHE_BOOL : False

# TRAINING_ARGS :
SEED : 56
DO_TRAIN : True
DO_EVAL : True
NUM_WORKERS : 4
BATCH_SIZE : 16
EPOCHS : 1
STEPS : 1
OPTIMIZER : 'adamw_torch'
LR : 1e-4
SCHEDULER_TYPE : 'linear'
WEIGHT_DECAY : 0.0
BETA1 : 0.9
BETA2 : 0.999
ADAM_EPS : 1e-8 
INTERVAL : 'epoch'
INTERVAL_STEPS : 100
NO_OF_CHECKPOINTS : 5
FP16 : False
RESUME_FROM_CHECKPOINT : False
GRADIENT_ACCUMULATION_STEPS : 1
GRADIENT_CHECKPOINTING : True
predict_with_generate: True
generation_max_length : 128
REMOVE_UNUSED_COLUMNS : True

# FINE_TUNING_ARGS :
LAST_LAYER_TUNING : True
FULL_FINE_TUNING : False

PEFT_METHOD : 'LoRA'

# LoRA_CONFIG :
r : 16
alpha : 8
dropout : 0.1
peft_type : 'LoRA'
target_modules : 
fan_in_fan_out : False
init_lora_weights : True  

# BNB_CONFIG :
load_in_4bit : True
bnb_4bit_compute_dtype : "float16"
bnb_4bit_quant_type : "nf4"
bnb_4bit_use_double_quant : False 
```

## Adapting the model
With the yaml file configured, the adaptation process is initiated with the following command : 

```bash 
nyun run examples/adapt/summarization/config.yaml
```

Once the job starts, you will find the following directory structure in the `user_data` folder:

```bash
user_data/
├── jobs
│   └── Adapt
│       └── SUMM
├── logs
    └── Adapt
        └── SUMM
            └── log.log

```
The output model will be stored in `user_data/jobs/Adapt/SUMM/` directory and the final directory structure will be:

```bash
user_data/
├── jobs
│   └── Adapt
│       └── SUMM
│           └── merged_model_state_dict.pth
├── logs
    └── Adapt
        └── SUMM
            └── log.log

```

## Conclusion 
This guide has walked you through the process of adapting the T5-Large model using QLoRA for summarization on the xsum dataset. By employing QLoRA, we efficiently fine-tuned the model with a reduced memory footprint.. The configuration and setup steps were outlined, ensuring that even complex tasks like distributed training and low-rank adaptation are manageable. The final trained model and logs are organized in a clear directory structure, making it easy to retrieve and analyze results.



---

*Author: [Panigrahi, Abhranta](https://www.linkedin.com/in/abhranta-panigrahi-626a23191/)*