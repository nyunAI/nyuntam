# Maximising math performance for extreme compressions: 2-bit Llama3-8b (w2a16)

## Overview

This guide provides a detailed walkthrough on maximizing the performance of a highly compressed Llama3-8b model using 2-bit weights and 16-bit activations. We will apply the Additive Quantization for Large Models (AQLM) technique to compress and optimize the Llama3-8b model, drastically reducing its memory footprint while maintaining performance.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Quantization](#running-the-quantization)
- [Performance Evaluation](#performance-evaluation)
- [Conclusion](#conclusion)

## Introduction

In this guide, we will try and maximise the Grade School Math abilities of an extremely compressed Llama3-8b model.

## Requirements

Before starting, ensure that you have the following:

- A GPU-enabled environment with CUDA support.
- The nyuntam repository cloned and set up as per the [Installation Guide](#installation).

## Installation

### Step 1: Clone the Nyuntam Repository

Clone the repository and navigate to the `nyuntam` directory:

```bash
git clone https://github.com/nyunAI/nyuntam.git
cd nyuntam
git submodule update --init text_generation
cd text_generation
git submodule update --init quantization/aqlm/AQLM
cd ..
```

### Step 2: Set Up the Environment

Create and activate an environment for the AQLM quantization:

```bash
conda create -n aqlm_quantization python=3.10 # or use virtualenv if preferred
conda activate aqlm_quantization
```

Install the required dependencies:

```bash
pip install torch==2.3.0 # (adjust version if needed)
pip install -r text_generation/quantization/aqlm/requirements.txt
```

## Experimentations

### Step 1: SFT + Iterative DPO

We first apply SFT + Iterative DPO to the model to boost upfront the downstream task performance. For a quicker reproducibility, we use the llama3 checkpoints provided by RLHFlow - `RLHFlow/LLaMA3-iterative-DPO-final` for this experiment.

### Step 2: AQLM Quantization

Dataset: We use the `openai/gsm8k` dataset for calibration and fine-tuning of the quantized model. Use the following script to create the dataset:

```bash
python examples/text-generation/aqlm_quantization/create_dataset.py
```

Next, we quantize and finetune the model.

## Configuration

Prepare the YAML configuration file specific to AQLM quantization. Use the following template as a starting point:

```yaml
# aqlm_quantization.yaml

# Model configuration
MODEL: "RLHFlow/LLaMA3-iterative-DPO-final" # this can be meta-llama/Meta-Llama-3.1-8B or any other model from huggingface

# Data configuration
DATASET_NAME: "togethercomputer/RedPajama-Data-1T-Sample"
TEXT_COLUMN: "text"                     
SPLIT: "train"

# Data configuration (if finetuning on gsm8k)
# DATASET_NAME: "gsm8k_restructured"
# DATA_PATH: "user_data/datasets/gsm8k_restructured"
# TEXT_COLUMN: "text"                     
# SPLIT: "train"

DATASET_SUBNAME: ""
FORMAT_STRING:

# Quantization configuration

llm:
  AQLM:
    # Quantization parameters
    save_intermediate_results: true
    dtype: "float16"
    overwrite: false

    # ...other params

# Job configuration
CUDA_ID: "0,1,2,3"
ALGORITHM: "AQLM"
JOB_SERVICE: "Kompress"
USER_FOLDER: "user_data"
JOB_ID: "aqlm_quantization"
CACHE_PATH: "user_data/.cache"
JOB_PATH: "user_data/jobs/aqlm_quantization"
LOGGING_PATH: "user_data/logs/aqlm_quantization"
ALGO_TYPE: "llm"
TASK: "llm"
```

## Running the Quantization
<!-- runs with a different dataset for calib -->

```bash
python main.py --yaml_path examples/text-generation/aqlm_quantization/config.yaml
```

## Running the Finetuning
<!-- runs with a different dataset for finetuning -->

With your YAML file configured, initiate the quantization process by running:

```bash
torchrun --nproc-per-node=4 main.py --yaml_path examples/text-generation/aqlm_quantization/config.yaml
```

Monitor the process to ensure the quantization completes successfully.

Once the job starts, the following directory structure will be created in the `user_data` folder:

```bash
user_data/
├── datasets
│   ├── gsm8k_restructured
│   └── togethercomputer
│       └── RedPajama-Data-1T-Sample
├── jobs
│   └── Kompress
│       └── aqlm_quantization
│           └── tmp
│               ├── caliberation
│               └── tokenized_dataset
|               ...
├── logs
│   └── aqlm_quantization
└── models
    └── RLHFlow
        └── LLaMA3-iterative-DPO-final
```

The quantized model will be saved in the `user_data/jobs/Kompress/aqlm_quantization` directory:

```bash
user_data/
└── jobs
    └── Kompress
        └── aqlm_quantization
            ...
```

## Performance Evaluation

After quantization, evaluate the performance of the quantized model using the provided evaluation script:

```bash
pip install lm-eval

## ===== GSM8K Evaluation =====

# baseline gsm8k 5 shot evaluation
accelerate launch --no-python lm_eval --model hf \
  --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cache_dir=user_data/.cache \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size "auto" \
  --output_path user_data/evals/meta-llama_3.1-8b/base/gsm8k/

# Llama3* gsm8k 5 shot evaluation
accelerate launch --no-python lm_eval --model hf \
  --model_args pretrained=RLHFlow/LLaMA3-iterative-DPO-final,cache_dir=user_data/.cache \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size "auto" \
  --output_path user_data/evals/meta-llama_3.1-8b/sft+iterative_dpo/gsm8k/

# Llama3Q gsm8k 5 shot evaluation
python examples/text-generation/aqlm_quantization/evaluate.py \
  --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --quantized_model "user_data/jobs/Kompress/aqlm_quantization/tmp/caliberation/" \
  --tasks "gsm8k:5" \
  --results "user_data/evals/meta-llama_3.1-8b/Llama3Q" \
  --cache_dir "user_data/.cache"

# Llama3Q PV Tuned gsm8k 5 shot evaluation
python examples/text-generation/aqlm_quantization/evaluate.py \
  --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --quantized_model "user_data/jobs/Kompress/aqlm_quantization/tmp/converted/" \
  --tasks "gsm8k:5" \
  --results "user_data/evals/meta-llama_3.1-8b/Llama3Q_PV_Tuned" \
  --cache_dir "user_data/.cache"

## ===== Preplexity Evaluation =====

# Llama3* perplexity evaluation
python examples/text-generation/aqlm_quantization/evaluate.py \
  --base_model "RLHFlow/LLaMA3-iterative-DPO-final" \
  --tasks "gptq_wikitext:0" \
  --results "user_data/evals/meta-llama_3.1-8b/Llama3*" \
  --cache_dir "user_data/.cache"

# baseline & Llama3Q perplexity evaluation
python examples/text-generation/aqlm_quantization/evaluate.py \
  --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --quantized_model "user_data/jobs/Kompress/aqlm_quantization/tmp/caliberation/" \
  --tasks "gptq_wikitext:0" \
  --results "user_data/evals/meta-llama_3.1-8b" \
  --cache_dir "user_data/.cache"

# Llama3Q PV Tuned perplexity evaluation
python examples/text-generation/aqlm_quantization/evaluate.py \
  --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --quantized_model "user_data/jobs/Kompress/aqlm_quantization/tmp/converted/" \
  --tasks "gptq_wikitext:0" \
  --results "user_data/evals/meta-llama_3.1-8b/Llama3Q_PV_Tuned" \
  --cache_dir "user_data/.cache"
```

Compare the results with the original model to assess the impact of quantization on accuracy and inference speed.

|                	| Llama3<br>(Llama3-8b) 	| Llama3*<br>(SFT + Iterative DPO Llama3) 	| Llama3Q PV Tuned<br>(Quantized + PV Tuned Llama3*) 	|
|----------------	|-----------------------	|-----------------------------------------	|----------------------------------------------------	|
| GSM8K (5 shot) 	| 50.9                  	| 78.99                                   	| 58.9                                               	|

## Conclusion

From the results, we can see that the Llama3Q PV Tuned model achieves a GSM8K score of 58.9, which is a significant improvement over the baseline Llama3-8b model. The model has been compressed to 2-bit weights and 16-bit activations, reducing its memory footprint while maintaining performance.

---

*Author: [Kushwaha, Shubham](https://www.linkedin.com/in/shwoobham/)*

### Additional Examples

- **[Efficient 4-bit Quantization (w4a16) of Llama3.1-8b for Optimized Text Generation](../awq_quantization/readme.md)**
- **[Llama3.1 70B: 0.5x the cost & size](../flap_pruning/readme.md)**
- **[Achieving Up to 2.5x TensorRTLLM Speedups: Efficient 4-8-4 Quantization (w4a8kv4) of Llama3.1-8b](../lmquant_quantization/readme.md)**
- **[Accelerating a 4-bit Quantised Llama Model](../tensorrtllm_engine/readme.md)**

---
