# Achieving Up to 2.5x TensorRTLLM Speedups: Efficient 4-8-4 Quantization (w4a8kv4) of Llama3.1-8b

## Overview

This guide provides a detailed walkthrough of applying **LMQuant** using the QoQ algorithm (quattuor-octo-quattuor) to quantize the Llama3.1-8b model. By using 4-bit weights, 8-bit activations, and 4-bit key-value cache (W4A8KV4), LMQuant aims to significantly reduce model size while maintaining high performance and efficient inference speed. This process is particularly beneficial for deploying large language models in environments with limited resources.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Quantization](#running-the-quantization)
- [Performance Evaluation](#performance-evaluation)
- [Conclusion](#conclusion)

## Introduction

In this example, we will use the QoQ algorithm provided by LMQuant to quantize the Llama3.1-8b model. QoQ (quattuor-octo-quattuor) utilizes W4A8KV4 quantization, a method that effectively compresses the model without sacrificing significant performance, making it suitable for deployment on both edge devices and large-scale servers.

## Requirements

Before starting, ensure that you have the following:

- A GPU-enabled environment with CUDA support.
- The LMQuant repository cloned and set up as described in the [Installation Guide](#installation).

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
conda create -n lmquant_quantization python=3.10 # or use virtualenv if preferred
conda activate lmquant_quantization
```

Install the required dependencies:

```bash
pip install torch==2.3.0 # (any other version as suitable)
pip install -r text_generation/engines/mit_han_lab_qserve/requirements.txt text_generation/engines/mit_han_lab_qserve/QServe/kernels # for QServe
pip install -r text_generation/quantization/mit_han_lab_lmquant/requirements.txt
```

## Configuration

Prepare the YAML configuration file specific to the QoQ quantization. Use the following template as a starting point:

```yaml
# lmquant_quantization.yaml

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
  LMQuant:
    # Quantization parameters
    save_model: True
    keep_scales: True
    loads_with_qserve: False
    dtype: float32

    quant_type: "gchn"
    quant.develop_dtype: torch.float32
    quant.smooth.xw.alpha: 0.05 
    quant.smooth.xw.beta: 0.95 
    quant.smooth.yx.strategy: GridSearch 
    quant.smooth.yx.beta: " -2"

    quant.wgts.calib_range.outputs_device: cpu
    quant.reorder.outputs_device: cpu
    quant.smooth.xw.outputs_device: cpu
    quant.smooth.yx.outputs_device: cpu

    # Nested dictionary for quantization parameters
    eval.tasks: ["wikitext", "arc_challenge"]
    eval.max_seq_length: 4096
    eval.evaluator: "lm_eval"


# Job configuration
CUDA_ID: "0,1,2,3"
ALGORITHM: "LMQuant"
JOB_SERVICE: "Kompress"
USER_FOLDER: "user_data"
JOB_ID: "lmquant_quantization"
CACHE_PATH: "user_data/.cache"
JOB_PATH: "user_data/jobs/lmquant_quantization"
LOGGING_PATH: "user_data/logs/lmquant_quantization"
ALGO_TYPE: "llm"
TASK: "llm"
```

## Running the Quantization

With your YAML file configured, initiate the quantization process by running:

```bash
python main.py --yaml_path examples/text-generation/lmquant_quantization/config.yaml
```

Monitor the process to ensure that the quantization completes successfully.

The output model will be saved in the `user_data/jobs/qoq_quantization` directory.

```bash
user_data/
└── jobs
    └── QServe
        └── qoq_quantization
            ├── config.json
            ├── generation_config.json
            ├── model.safetensors
            ├── special_tokens_map.json
            ├── tokenizer.json
            └── tokenizer_config.json
```

## Performance Evaluation

After quantization, evaluate the performance of the quantized model using the evaluation script provided.

```bash
pip install lm-eval git+https://github.com/PanQiWei/AutoGPTQ.git

# wikitext perplexity evaluation
python examples/text-generation/qoq_quantization/evaluate.py \
  --base_model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --quantized_model "user_data/jobs/QServe/qoq_quantization" \
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
  --model_args pretrained=user_data/jobs/QServe/qoq_quantization,cache_dir=user_data/.cache \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size 16 \
  --output_path user_data/evals/meta-llama_3.1-8b/quantized/arc_challenge/
```

Compare the results with the original model to assess the impact of quantization on accuracy and inference speed.

### Performance Metrics

| Model        | Task                         | Metric        | Baseline  | Quantized  | Impact  |
|------------- |----------------------------- |-------------- |---------- |----------- |-------- |
| Llama3.1-8b  | Perplexity (wikitext, gptq)  | Perplexity ↓  | 6.14      | 6.76       | +0.62   |
| Llama3.1-8b  | ARC Challenge (25 shot)      | Accuracy ↑    | 60.66     | 60.10      | -0.56   |
| Llama3.1-8b  | Size (in GB)                 | Model Size ↓  | 15.0      | 05.1       | -09.90  |

### Throughput Comparison

| A100 (80G)  | TRT-LLM-FP16  | TRT-LLM-W4A16  | TRT-LLM-W8A8  | QServe-W4A8KV4  | Throughput Increase*  |
|------------ |-------------- |--------------- |-------------- |---------------- |---------------------- |
| Llama-3-8B  | 2503          | 2370           | 2396          | **3005**        | **1.20x**             |

*

## Conclusion

LMQuant’s QoQ algorithm offers a robust method for compressing large models like Llama3.1-8b, balancing performance with efficiency. By following this guide, you should now have a quantized model ready for deployment in resource-constrained environments.

---

*Author: [Kushwaha, Shubham](https://www.linkedin.com/in/shwoobham/)*

### Additional Examples

- **[Maximising math performance for extreme compressions: 2-bit Llama3-8b (w2a16)](../aqlm_quantization/readme.md)**
- **[Efficient 4-bit Quantization (w4a16) of Llama3.1-8b for Optimized Text Generation](../awq_quantization/readme.md)**
- **[Llama3.1 70B: 0.5x the cost & size](../flap_pruning/readme.md)**
- **[Accelerating a 4-bit Quantised Llama Model](../tensorrtllm_engine/readme.md)**

---
