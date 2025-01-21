# Accelerating a 4-bit Quantised Llama Model

## Overview

This guide demonstrates how to accelerate a 4-bit quantized Llama model using the TensorRTLLM engine. TensorRTLLM is a high-performance inference engine that leverages NVIDIA's TensorRT library to optimize and accelerate models for deployment on NVIDIA GPUs.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the engine build](#running-the-engine-build)
- [Performance Evaluation](#performance-evaluation)
- [Conclusion](#conclusion)

## Introduction

In this example, we'll demonstrate how to accelerate a 4-bit quantized model using the TensorRTLLM engine. The process involves quantizing the model using the AWQ quantization technique and then optimizing it for deployment on NVIDIA GPUs using TensorRTLLM.

## Requirements

Before you begin, ensure that you have the following:

- A GPU-enabled environment with CUDA support.

## Installation

### Step 1: Clone the Nyuntam Repository

Clone the repository and navigate to the `nyuntam` directory:

```bash
git clone https://github.com/nyunAI/nyuntam.git
cd nyuntam/examples/text-generation/tensorrtllm_engine/
```

### Step 2: Set Up the workspace

Create and activate an environment for the AWQ quantization example:

```bash
conda create -n tensorrtllm_engine python=3.10 -y # or use virtualenv if preferred
conda activate tensorrtllm_engine
```

Install the required dependencies:

```bash
pip install git+https://github.com/nyunAI/nyunzero-cli.git
```

Setup the nyun workspace

```bash
mkdir workspace && cd workspace
nyun init -e kompress-text-generation # wait for the extensions to be installed
```

## Configuration

Prepare the YAML configuration file specific to AWQ quantization. Use the following template as a starting point:

```yaml
# tensorrtllm_engine.yaml

# Model configuration
MODEL: "meta-llama/Llama-2-7b-hf"

# Data configuration
DATASET_NAME: "wikitext"
DATASET_SUBNAME: "wikitext-2-raw-v1"
TEXT_COLUMN: "text"                     
SPLIT: "train"

DATA_PATH:
FORMAT_STRING:

# Acceleration configuration
llm:
  TensorRTLLM:
    to_quantize: true # to first quantize the model and then build engine. (Supported only for llama, gptj, & falcon models.)
    dtype: float16

    # quantization parameters
    quant_method: "int4_awq" # 'fp8', 'int4_awq', 'smoothquant', 'int8'
    smoothquant: 0.5 # in case smoothquant value is given
    calib_size: 32

    ...other params

# Job configuration
CUDA_ID: "0"
ALGORITHM: "TensorRTLLM"
JOB_SERVICE: "Kompress"
USER_FOLDER: "/user_data/example"
JOB_ID: "tensorrtllm_engine"
CACHE_PATH: "/user_data/example/.cache"
JOB_PATH: "/user_data/example/jobs/tensorrtllm_engine"
LOGGING_PATH: "/user_data/example/logs/tensorrtllm_engine"
ALGO_TYPE: "llm"
TASK: "llm"
```

## Running the engine build

With your YAML file configured, initiate the process by running:

```bash
nyun run ../config.yaml
```

Monitor the process to ensure that the quantization completes successfully.

Once the job starts, you'll find the following directory structure in the `workspace` folder:

```bash
workspace/
├── custom_data
└── example
    ├── datasets
    │   └── wikitext
    ├── jobs
    │   └── Kompress
    │       └── tensorrtllm_engine
    ├── logs
    │   └── tensorrtllm_engine
    └── models
        └── meta-llama
            └── Llama-2-7b-hf
                ...
```

The output model will be saved in the `workspace/example/jobs/Kompress/tensorrtllm_engine` directory.

## Performance Evaluation

Following is the comparison of the results* with the original model to assess the impact of quantization on accuracy and inference speed.

| Model                     | Optimised with             | Quantization Type                      | WM (GB)  | RM (GB)  | Tokens/s  | Perplexity  |
|-------------------------- |--------------------------- |--------------------------------------- |--------- |--------- |---------- |------------ |
| meta-llama/Llama-2-7b-hf  | TensorRT-LLM               | AWQ GEMM 4bit (quant_method=int4_awq)  | 3.42     | 5.69     | 194.86    | 6.02        |
|                           |                            | INT8 (quant_method=int8)               | 6.53     | 8.55     | 143.57    | 5.89        |
|                           |                            | FP16 (to_quantize=false)               | 12.55    | 14.61    | 83.43     | 5.85        |
| meta-llama/Llama-2-7b-hf  | Text-Generation-Inference  | AWQ GEMM 4bit                          | 3.62     | 36.67    | 106.84    | 6.02        |
|                           |                            | FP16                                   | 12.55    | 38.03    | 74.19     | 5.85        |

_*Source: [Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward](https://arxiv.org/abs/2402.01799)_

## Conclusion

In this example, we demonstrated how to accelerate a 4-bit quantized Llama3.1-8b model using the TensorRTLLM engine. By leveraging the nyun cli, we optimized the model for deployment on NVIDIA GPUs, achieving significant improvements in inference speed and memory efficiency.

---

_Author: [Kushwaha, Shubham](https://www.linkedin.com/in/shwoobham/)_

### Additional Examples

- **[Maximising math performance for extreme compressions: 2-bit Llama3-8b (w2a16)](../aqlm_quantization/readme.md)**
- **[Efficient 4-bit Quantization (w4a16) of Llama3.1-8b for Optimized Text Generation](../awq_quantization/readme.md)**
- **[Llama3.1 70B: 0.5x the cost & size](../flap_pruning/readme.md)**
- **[Achieving Up to 2.5x TensorRTLLM Speedups: Efficient 4-8-4 Quantization (w4a8kv4) of Llama3.1-8b](../lmquant_quantization/readme.md)**

---
