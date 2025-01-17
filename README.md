# Nyuntam

**Nyuntam** is NyunAI's advanced suite designed to optimize and accelerate large language models through state-of-the-art compression techniques, featuring an integrated CLI for seamless workflow management.

## Key Features

- **State-of-the-Art Compression**: Includes techniques such as pruning, quantization, and distillation to ensure model efficiency without compromising performance
- **Multi-Platform Support**: Run experiments seamlessly on various platforms using Docker or virtual environments
- **Integrated CLI**: Built-in command-line interface for easy workspace management and experiment execution
- **Extensible Architecture**: Support for various tasks including text generation, computer vision, and model adaptation

## Installation

### Prerequisites

- Python 3.6 or later
- For GPU support: NVIDIA Container Toolkit (when using Docker)

### Quick Install

Install Nyuntam using pip:

```bash
pip install nyuntam
```

### Alternative Installation Methods

#### Git + Docker

1. **Install NVIDIA Container Toolkit** (Linux):
    ```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

2. **Clone and Setup**:
    ```bash
    git clone --recursive https://github.com/nyunAI/nyuntam.git
    cd nyuntam
    docker pull nyunadmin/nyuntam-text-generation:latest
    docker run -it -d --gpus all -v $(pwd):/workspace --name nyuntam-dev --network=host nyunadmin/nyuntam-text-generation:latest bash
    ```

#### Git + Virtual Environment

1. **Clone Repository**:
    ```bash
    git clone --recursive https://github.com/nyunAI/nyuntam.git
    cd nyuntam
    ```

2. **Setup Environment**:
    ```bash
    python3 -m venv {ENVIRONMENT_NAME}
    source {ENVIRONMENT_NAME}/bin/activate
    pip install -r requirements.txt
    ```

#### Using Setup.py

```bash
python install . [extra_deps]
```

Available extra dependencies:
- **Text Generation**
  - "flap": For running Flap
  - "aqlm": For running AQLM
  - "autoawq": For running AutoAWQ quantization
  - "qserve": For running QServe

## Usage

### Initializing a Workspace

Before running experiments, initialize your workspace:

```bash
nyuntam init [WORKSPACE_PATH] [CUSTOM_DATA_PATH] [OPTIONS]
```

Options:
- `--overwrite`, `-o`: Overwrite existing workspace
- `--extensions`, `-e`: Specify extensions to install:
  - `kompress-vision`: For vision tasks
  - `kompress-text-generation`: For text generation
  - `adapt`: For model adaptation
  - `all`: Install all extensions
  - `none`: No extensions

Example:
```bash
nyuntam init ~/my-workspace ~/my-data --extensions kompress-vision
```

### Running Experiments

1. **Prepare Configuration**:
   Create a YAML file defining your experiment parameters. Example configurations are available in the `scripts` directory.

   - Refer to [dataset imports](https://nyunai.github.io/nyuntam-docs/dataset/) and [models imports](https://nyunai.github.io/nyuntam-docs/model/) for configuration details.
   - Scripts and example YAML files are available [here](https://github.com/nyunAI/nyuntam-text-generation/tree/main/scripts).

2. **Execute**:
   ```bash
   nyuntam run path/to/recipe.yaml
   ```

   For chained execution:
   ```bash
   nyuntam run script1.yaml script2.yaml
   ```

## Examples

For detailed examples and use cases, refer to our [examples directory](./nyuntam/examples/readme.md), which includes:

### Text Generation
- [Maximising math performance for extreme compressions: 2-bit Llama3-8b (w2a16)](./nyuntam/examples/text-generation/aqlm_quantization/readme.md)
- [Efficient 4-bit Quantization (w4a16) of Llama3.1-8b](./nyuntam/examples/text-generation/awq_quantization/readme.md)
- [Llama3.1 70B: 0.5x the cost & size](./nyuntam/examples/text-generation/flap_pruning/readme.md)
- [Achieving Up to 2.5x TensorRTLLM Speedups](./nyuntam/examples/text-generation/lmquant_quantization/readme.md)
- [Accelerating a 4-bit Quantised Llama Model](./nyuntam/examples/text-generation/tensorrtllm_engine/readme.md)

### Computer Vision
- [Pruning YOLOX with MMRazor](./nyuntam/examples/vision/mmrazor_pruning/readme.md)
- [8-bit CPU Quantization of ResNet50](./nyuntam/examples/vision/nncf_quantization/readme.md)

### Model Adaptation
- [Adapting SegNeXt to cityscapes dataset using SSF](./nyuntam/examples/adapt/image_segmentation/README.md)
- [Finetuning RTMDet on face-det dataset using LoRA and DDP](./nyuntam/examples/adapt/object_detection/README.md)
- [Finetuning T5 large with QLoRA on XSUM dataset](./nyuntam/examples/adapt/summarization/README.md)
- [Finetuning Llama3-8b with QDoRA and FSDP](./nyuntam/examples/adapt/text_generation/README.md)

## Documentation

For complete documentation, visit [NyunAI Docs](https://nyunai.github.io/nyuntam-docs)

## Version Information

Check your installed version:
```bash
nyuntam version
```

---

<span style="color:red">**NOTE:**</span> For access to gated repositories within containers, ensure you have the necessary Hugging Face tokens configured.