# Finetuning Llama3-8b with QDoRA and FSDP

## Table on Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Adapting the model](#adapting-the-model)
- [Conclusion](#conclusion)

## Introduction

In this example we will be finetuning Llama3-8b with QDoRA and FSDP. We will be using the the Llama-1k dataset for this example but any dataset (properly formatted) can be used for this purpose. [DoRA](https://arxiv.org/abs/2402.09353) is a PEFT method which simillar to LoRA trains adapter by freezing and quantizing the original model weights.

## Requirements

Before you begin, ensure that you have the following:

- A GPU-enabled environment with CUDA support.
- The Nyuntam repository cloned and set up as per the [Installation Guide](#installation).
- Docker

## Installation

### Step 1: Clone the Nyuntam Repository

Clone the repository and navigate to the `nyuntam` directory:

```bash
git clone https://github.com/nyunAI/nyuntam.git
cd nyuntam
```

### Step 2: Set Up the workspace

To setup the environment use the following command(s),

```bash
pip install git+https://github.com/nyunAI/nyunzero-cli.git
nyun init {WORKSPACE_PATH} -e adapt
```

## Dataset

For this example we are using the [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) dataset from 🤗 huggingface. Other datasets might need proper formatting according to the model and the task.

| Text                                                                                                                                                                                                                                       |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **[INST]** Me gradué hace poco de la carrera de medicina ¿Me podrías aconsejar para conseguir rápidamente un puesto de trabajo? **[/INST]** Esto vale tanto para médicos como para cualquier otra profesión tras finalizar los...          |
| **[INST]** Самый великий человек из всех живших на планете? **[/INST]** Для начала нужно выбрать критерии величия человека. Обычно великим называют человека, который внес большой вклад в общество или сильно выделялся на фоне...          |
| **[INST]** Compose a professional email with the following points: Me chinese cook 10 years Good good cook People love my food Can you hire me? **[/INST]** Thanks for the information. Unfortunately, your initial message...             |
| **[INST]** ¿Qué juegos me recomendarías si me ha gustado Undertale? **[/INST]** Deltarune: Este es un juego creado por el mismo desarrollador de Undertale, Toby Fox. Al igual que Undertale, Delt

## Configuration

The configuration yaml file is as follows:

```yaml
#Training Arguments
ADAM_EPS: 1e-8
BATCH_SIZE: 4
BETA1: 0.9
BETA2: 0.999
CACHE_BOOL: false
DO_EVAL: false
DO_TRAIN: true
EPOCHS: 0.01
GRADIENT_ACCUMULATION_STEPS: 1
GRADIENT_CHECKPOINTING: true
GROUP_BY_LENGTH: true
INTERVAL: steps
INTERVAL_STEPS: 50
LR: 0.0002
NO_OF_CHECKPOINTS: 5
NUM_WORKERS: 4
OPTIMIZER: paged_adamw_32bit
REMOVE_UNUSED_COLUMNS: true
RESUME_FROM_CHECKPOINT: false

#Dataset Arguments
CUSTOM_DATASET_PATH: null
DATASET: mlabonne/guanaco-llama2-1k
DATASET_CONFIG: {}
DATASET_FORMAT: null
DATASET_ID: 27
FP16: true
SCHEDULER_TYPE: constant
SEED: 56
STEPS: 100
WEIGHT_DECAY: 0.001
alpha: 16
blocksize: 128
dataset_text_field: text
max_seq_length: 512
packing: true

#Model Arguments
LOCAL_MODEL_PATH: null
MODEL: Llama-2
MODEL_PATH: NousResearch/Llama-2-7b-hf

#Basic Arguments
ID: 27
JOB_ID: 100
JOB_SERVICE: Adapt
LOGGING_PATH: /user_data/logs/Adapt/100
OUTPUT_DIR: /user_data/jobs/Adapt/100
OVERWRITE_OUTPUT_DIR: false
Library: Huggingface
FSDP: true
SAVE_METHOD: state_dict
TASK: text_generation
USER_FOLDER: user_data
cuda_id: '0,1,2,3'
num_nodes: 1

#PEFT arguments
FULL_FINE_TUNING: false
LAST_LAYER_TUNING: true
PEFT_METHOD: DoRA
dropout: 0.1
fan_in_fan_out: false
flash_attention2: false
init_lora_weights: true
peft_type: DoRA
r: 2
target_modules: null
auto_select_modules: true

#Quantization Arguments
load_in_4bit: true
bnb_4bit_compute_dtype: float16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: false

#FSDP configs
compute_environment: LOCAL_MACHINE
debug: true
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
fsdp_backward_prefetch: NO_PREFETCH
fsdp_cpu_ram_efficient_loading: true
fsdp_forward_prefetch: false
fsdp_offload_params: false
fsdp_sharding_strategy: FULL_SHARD
fsdp_state_dict_type: SHARDED_STATE_DICT
fsdp_sync_module_states: true
fsdp_transformer_layer_cls_to_wrap: null
fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: true
```

## Adapting the model

With the yaml file configured, the adaptation process is initiated with the following command:

```bash
nyun run examples/adapt/text_generation/config.yaml
```

Once the job starts, you will find the following directory structure in the `user_data` folder:

```bash
user_data/
├── jobs
│   └── Adapt
│       └── 100
├── logs
    └── Adapt
        └── 100
            └── log.log

```

The output model will be stored in `user_data/jobs/Adapt/100/` directory and the final directory structure will be:

```bash
user_data/
├── jobs
│   └── Adapt
│       └── 100
│           └── merged_model_state_dict.pth
├── logs
    └── Adapt
        └── 100
            └── log.log

```

## Conclusion

This guide has walked you through the process of adapting the Llama3-8b model using QDoRA for text generation. By employing QDoRA, we efficiently fine-tuned the model with a reduced memory footprint. The configuration and setup steps were outlined, ensuring that even complex tasks like distributed training and low-rank adaptation are manageable. The final trained model and logs are organized in a clear directory structure, making it easy to retrieve and analyze results.

---

*Author: [Panigrahi, Abhranta](https://www.linkedin.com/in/abhranta-panigrahi-626a23191/)*
