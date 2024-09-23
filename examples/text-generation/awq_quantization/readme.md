# Efficient 4-bit Quantization (w4a16) of Llama3.1-8b for Optimized Text Generation

## Overview

This guide provides a walkthrough of applying **AWQ** (Activation-aware Weight Quantization) to compress and accelerate the Llama3.1-8b model using 4-bit weights and 16-bit activations. AWQ allows for significant reduction in model size and computational requirements without sacrificing performance, making it an excellent choice for deploying large language models in resource-constrained environments.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Quantization](#running-the-quantization)
- [Performance Evaluation](#performance-evaluation)
- [Conclusion](#conclusion)

## Introduction

In this example, we'll be utilizing the AWQ technique to quantize the Llama3.1-8b model, aiming to reduce its memory footprint and improve inference speed. AWQ's flexibility in precision makes it a versatile tool for adapting large models to various hardware configurations.

## Requirements

Before you begin, ensure that you have the following:

- A GPU-enabled environment with CUDA support.
- The Nyuntam repository cloned and set up as per the [Installation Guide](#installation).

## Installation

### Step 1: Clone the Nyuntam Repository

Clone the repository and navigate to the `nyuntam` directory:

```bash
git clone https://github.com/nyunAI/nyuntam.git
cd nyuntam
```

### Step 2: Set Up the workspace

Create and activate an environment for the AWQ quantization example:

```bash
conda create -n awq_quantization python=3.10 # or use virtualenv if preferred
conda activate awq_quantization
```

Install the required dependencies:

```bash
pip install torch==2.3.0 # (any other version as suitable)
pip install -r text_generation/quantization/autoawq/requirements.txt
```

## Configuration

Prepare the YAML configuration file specific to AWQ quantization. Use the following template as a starting point:

```yaml
# awq_quantization.yaml

# Model configuration
MODEL: "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Data configuration
DATASET_NAME: "wikitext"
DATASET_SUBNAME: "wikitext-2-raw-v1"
TEXT_COLUMN: "text"                     
SPLIT: "train"

DATA_PATH:
FORMAT_STRING:

# Quantization configuration
llm:
  AutoAWQ:
    ZERO_POINT: True                    # zero point quantization
    W_BIT: 4                            # weight bitwidth
    Q_GROUP_SIZE: 128                   # group size for quantization [default: 128, 64, 32]
    VERSION: "GEMV"                     # quantization version (GEMM or GEMV)

# Job configuration
CUDA_ID: "0"
ALGORITHM: "AutoAWQ"
JOB_SERVICE: "Kompress"
USER_FOLDER: "user_data"
JOB_ID: "awq_quantization"
CACHE_PATH: "user_data/.cache"
JOB_PATH: "user_data/jobs/awq_quantization"
LOGGING_PATH: "user_data/logs/awq_quantization"
ALGO_TYPE: "llm"
TASK: "llm"
```

## Running the Quantization

With your YAML file configured, initiate the quantization process by running:

```bash
python main.py --yaml_path examples/text-generation/awq_quantization/config.yaml
```

Monitor the process to ensure that the quantization completes successfully.

Once the job starts, you'll find the following directory structure in the `user_data` folder:

```bash
user_data/
├── datasets
│   └── wikitext
├── jobs
│   └── Kompress
│       └── awq_quantization
├── logs
│   └── awq_quantization
└── models
    └── meta-llama
        └── Meta-Llama-3.1-8B-Instruct
```

The output model will be saved in the `user_data/jobs/Kompress/awq_quantization` directory.

```bash
user_data/
└── jobs
    └── Kompress
        └── awq_quantization
            ├── config.json
            ├── generation_config.json
            ├── model.safetensors
            ├── special_tokens_map.json
            ├── tokenizer.json
            └── tokenizer_config.json
```

## Performance Evaluation

After the quantization process, evaluate the performance of the quantized model using the evaluation script provided.

```bash
pip install lm-eval git+https://github.com/PanQiWei/AutoGPTQ.git

# wikitext perplexity evaluation
python examples/text-generation/awq_quantization/evaluate.py \
  --base_model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --quantized_model "user_data/jobs/Kompress/awq_quantization" \
  --tasks "gptq_wikitext:0" \
  --results "user_data/evals/meta-llama_3.1-8b" \
  --cache_dir "user_data/.cache"

# baseline arc_challenge 25 shot evaluation
accelerate launch --no-python lm_eval --model hf \
  --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,cache_dir=user_data/.cache \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size 16 \
  --output_path user_data/evals/meta-llama_3.1-8b/base/arc_challenge/

# quantized arc_challenge 25 shot evaluation
accelerate launch --no-python lm_eval --model hf \
  --model_args pretrained=user_data/jobs/Kompress/awq_quantization,cache_dir=user_data/.cache \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size 16 \
  --output_path user_data/evals/meta-llama_3.1-8b/quantized/arc_challenge/
```

Compare the results with the original model to assess the impact of quantization on accuracy and inference speed.

| Model        | Task                         | Metric        | Baseline  | Quantized  | Impact  |
|------------- |----------------------------- |-------------- |---------- |----------- |-------- |
| Llama3.1-8b  | Perplexity (wikitext, gptq)  | Perplexity ↓  | 7.32      | 7.69       | +0.37   |
| Llama3.1-8b  | ARC Challenge (25 shot)      | Accuracy ↑    | 60.66     | 59.73      | -0.93   |
| Llama3.1-8b  | Size (in GB)                 | Model Size ↓  | 15.0      | 05.4       | -09.60  |

## Conclusion

AWQ quantization offers a practical approach to optimizing large models like meta-llama/Meta-Llama-3.1-8B-Instruct for deployment in limited-resource environments. By following this guide, you should now have a quantized model that balances performance and efficiency.

---

*Author: [Kushwaha, Shubham](https://www.linkedin.com/in/shwoobham/)*

### Additional Examples

- **[Maximising math performance for extreme compressions: 2-bit Llama3-8b (w2a16)](../aqlm_quantization/readme.md)**
- **[Llama3.1 70B: 0.5x the cost & size](../flap_pruning/readme.md)**
- **[Achieving Up to 2.5x TensorRTLLM Speedups: Efficient 4-8-4 Quantization (w4a8kv4) of Llama3.1-8b](../lmquant_quantization/readme.md)**
- **[Accelerating a 4-bit Quantised Llama Model](../tensorrtllm_engine/readme.md)**

---
