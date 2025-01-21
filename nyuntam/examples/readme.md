# Fast, Small and Efficient AI
>
> _At NyunAI, we are committed to developing fast, small, and efficient AI models that can be deployed in datacenters, inference-servers & edge devices with limited resources. Our examples demonstrate various compression and adaption techniques for Deep Learning models, focusing on reducing model size and improving inference speed while maintaining accuracy._

This directory contains a collection of examples demonstrating multiple techniques applied on various Computer Vision, NLP, and GenAI models.

---

## Text Generation

### 1. [Maximising math performance for extreme compressions: 2-bit Llama3-8b (w2a16)](./text-generation/aqlm_quantization/readme.md)

_This guide provides a detailed walkthrough on maximizing the performance of a highly compressed Llama3-8b model using 2-bit weights and 16-bit activations. We will apply the Additive Quantization for Large Models (AQLM) technique to compress and optimize the Llama3-8b model, drastically reducing its memory footprint while maintaining performance._

### 2. [Efficient 4-bit Quantization (w4a16) of Llama3.1-8b for Optimized Text Generation](./text-generation/awq_quantization/readme.md)

_This guide provides a walkthrough of applying **AWQ** (Activation-aware Weight Quantization) to compress and accelerate the Llama3.1-8b model using 4-bit weights and 16-bit activations. AWQ allows for significant reduction in model size and computational requirements without sacrificing performance, making it an excellent choice for deploying large language models in resource-constrained environments._

### 3. [Llama3.1 70B: 0.5x the cost & size](./text-generation/flap_pruning/readme.md)

_This guide provides a walkthrough of applying FLAP (Fluctuation-based Adaptive Structured Pruning) to compress and accelerate the Llama3.1-70b model. FLAP allows for significant reduction in model size and computational requirements without sacrificing performance. Unlike traditional pruning techniques, FLAP requires no retraining and adapts the pruning ratio across different modules and layers, offering an efficient and effective approach for deploying large language models in resource-constrained environments._

### 4. [Achieving Up to 2.5x TensorRTLLM Speedups: Efficient 4-8-4 Quantization (w4a8kv4) of Llama3.1-8b](./text-generation/lmquant_quantization/readme.md)

_This guide provides a detailed walkthrough of applying LMQuant using the QoQ algorithm (quattuor-octo-quattuor) to quantize the Llama3.1-8b model. By using 4-bit weights, 8-bit activations, and 4-bit key-value cache (W4A8KV4), LMQuant aims to significantly reduce model size while maintaining high performance and efficient inference speed. This process is particularly beneficial for deploying large language models in environments with limited resources._

### 5. [Accelerating a 4-bit Quantised Llama Model](./text-generation/tensorrtllm_engine/readme.md)

_This guide demonstrates how to accelerate a 4-bit quantized (awq) Llama model with the TensorRTLLM engine. TensorRTLLM is a high-performance inference engine that leverages NVIDIA's TensorRT library to optimize and accelerate models for deployment on NVIDIA GPUs._

## Computer Vision

### 1. [Pruning YOLOX with MMRazor](./vision/mmrazor_pruning/readme.md)

_Pruning Object Detection is supported via MMRazor Pruning. Currently activation based pruning and flop based pruning are the available option. This example utilizes NyunCLI a quick and seamless solution to run the pruning job._

### 2. [8-bit CPU Quantization of ResNet50 using NNCF on CIFAR-10 Dataset](./vision/nncf_quantization/readme.md)

_8-bit quantization of Image Classification models are done via NNCF, ONNX Quantizers for CPU and TensorRT Quantizer for GPU deployment. This example shows how to Quantize a ResNet50 model with NNCF for CPU Quantization._

## Adaption

### 1. [Adapting SegNeXt to cityscapes dataset using SSF](./adapt/image_segmentation/README.md)

_This guide provides a walkthrough of applying SegNeXt for instance segmentation on the cityscapes dataset using SSF (Scaling and Shifting the deep Features). SSF enables parameter efficient fine-tuning by proposing that performace simillar to full fine-tuning can be achieved by only scaling and shifting the features of a deep neural network._

### 2. [Finetuning RTMDet on face-det dataset using LoRA and DDP](./adapt/object_detection/README.md)

_This guide provides a walkthrough of applying RTMDet for face detection on the face-det dataset using LoRA (Low-Rank Adaptation) with Distributed Data Parallel (DDP) across 2 GPUs. LoRA enables efficient fine-tuning by reducing the memory footprint, making it a powerful approach for high-performance face detection while maintaining scalability and resource efficiency._

### 3. [Finetuning T5 large with QLoRA on XSUM dataset](./adapt/summarization/README.md)

_This guide provides a detailed walkthrough for finetuning T5 Large model on the xsum Dataset with QLoRA using nyuntam-adapt. QLoRA is a PEFT technique where the original weights are frozen to reduce the trainable parameters and and qre quantized to reduce the memory usage._

### 4. [Finetuning Llama3-8b with QDoRA and FSDP](./adapt/text_generation/README.md)

_In this example we will be finetuning Llama3-8b with QDoRA and FSDP. We will be using the the Llama-1k dataset for this example but any dataset can be used for this purpose. [DoRA](https://arxiv.org/abs/2402.09353) is a PEFT method which, simillar to LoRA, trains adapters by freezing and quantizing the original model weights._

---
