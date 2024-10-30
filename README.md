# Nyuntam

**Nyuntam** is NyunAI's advanced suite designed to optimize, adapt, and accelerate a wide array of deep learning models across various domains. The repository is structured into several submodules, each targeting specific tasks:

- **Nyuntam Text Generation**: Focuses on compressing large language models for text generation tasks.
- **Nyuntam Vision**: Tailored for compressing and optimizing vision models.
- **Nyuntam Adapt**: A robust module for fine-tuning and transfer learning across both vision and language models leveraging SoTA PEFT, full-finetuning and GPU parallelism.

## Key Features

- **State-of-the-Art Compression**: Includes techniques such as pruning, quantization, distillation, and more, to ensure model efficiency without compromising performance.
- **Adaptation**: Fine-tune and adapt vision and language models to specific tasks and datasets using methods like (Q)LoRA, (Q)SSF, and others.
- **Multi-Platform Support**: Run experiments seamlessly on various platforms using Docker or virtual environments.
- **CLI Integration**: Simplify your workflow with the [NyunZero CLI](https://github.com/nyunAI/nyunzero-cli?tab=readme-ov-file#nyun-cli).

## Installation

### NyunZero CLI (Recommended)

The fastest way to install and use Nyuntam is through the NyunZero CLI. It streamlines the setup process and gets you running experiments in no time.

For more details, visit the [NyunZero CLI Documentation](https://github.com/nyunAI/nyunzero-cli).

### Git + Docker

1. **Clone the Repository**:

    ```bash
    git clone --recursive https://github.com/nyunAI/nyuntam.git
    cd nyuntam
    ```

2. **Pull the Required Docker Image**:

    ```bash
    docker pull <docker_image_name_or_id>
    ```

3. **Run the Docker Container**:

    ```bash
    docker run -it -d --gpus all -v $(pwd):/workspace --name {CONTAINER_NAME} --network=host <docker_image_name_or_id> bash
    ```

<span style="color:red">**NOTE:**</span> Ensure that `nvidia-container-toolkit` is installed before running the Docker container. For access to gated repositories within the containers, add Hugging Face tokens.

### Git + Virtual Environment

1. **Clone the Repository**:

    ```bash
    git clone --recursive https://github.com/nyunAI/nyuntam.git
    cd nyuntam
    ```

2. **Create and Activate a Virtual Environment**:

    ```bash
    python3 -m venv {ENVIRONMENT_NAME}
    source {ENVIRONMENT_NAME}/bin/activate
    ```

3. **Install the Required Packages**:

    ```bash
    # for text_generation
    pip install -r text_generation/requirements.txt 
    # for vision
    pip install -r vision/requirements.txt 
    # for nyuntam_adapt
    pip install -r nyuntam_adapt/requirements.txt 
    ```

### Docker for Submodules

For running each submodule independently via Docker, use the following commands:

- **Nyuntam Text Generation**:

    ```bash
    docker pull nyunadmin/nyunzero_text_generation:v0.1
    docker run -it -d --gpus all -v $(pwd):/workspace --name {CONTAINER_NAME} --network=host nyunadmin/nyunzero_text_generation:v0.1 bash
    ```

- **Nyuntam Vision**:

    ```bash
    docker pull nyunadmin/nyunzero_kompress_vision:v0.1 
    docker run -it -d --gpus all -v $(pwd):/workspace --name {CONTAINER_NAME} --network=host nyunadmin/nyunzero_kompress_vision:v0.1 bash
    ```

- **Nyuntam Adapt**:

    ```bash
    docker pull nyunadmin/nyunzero_adapt:v0.1
    docker run -it -d --gpus all -v $(pwd):/workspace --name {CONTAINER_NAME} --network=host nyunadmin/nyunzero_adapt:v0.1 bash
    ```
### Using Setup.py
    ```bash
    python install . [extra_deps]
    ```
The extra dependencies allow installation of individual components of Nyuntam, the extra_dependies according to submodules is listed below
- **Nyuntam Text Generation**
  - "flap": For running Flap. You should also add "adapt" extra dependencies with "flap".
  - "aqlm": For running aqlm.
  - "autoawq": For running autoawq quantization.
  - "qserve": For running qserve.
- **Nyuntam Vision**
  - "classification_stack": For complete installation of classification stack which includes Quantization, Distillation and Pruning.
  - "classification" For Installing Model and Dataset Support without compression support. 
  - "nncf": For NNCF QAT and PTQ. (CPU)
  - "tensorrt": For GPU Quantization of Classification Models.
  - "openvino": For Openvino CPU PTQ.
  - "torchprune": For Pruning Classification Models.
- **Nyuntam Adapt**
  - "adapt": For complete installation of Adapt.
  - "accelerate": For supporting multi-gpu training with Adapt.
- **Exceptions**
      OpenMM libraries and TensorRTLLM installations require manual installation.
  - **OpenMM Libraries**
          Nyuntam utilies OpenMM Libraires for importing and compressing object detection models. We utilize MMDetection and MMYolo for object detection models, MMSegmentation for           Segmentaion models, MMRazor and MMDeploy for compression and deployment of models. To install the proper libraries follow the code snippet below

          ```bash
          pip install openmim
          mim install mmdet mmseg mmyolo mmrazor mmdeploy
          ```
  
  - **TensorRTLLM**
      To install TensorRTLLM follow the official instructions found here: [Linux](https://nvidia.github.io/TensorRT-LLM/installation/linux.html) and [Windows](https://nvidia.github.io/TensorRT-LLM/installation/windows.html).
## Usage

### Setting Up YAML Configuration Files

Each submodule in Nyuntam requires a YAML file that defines all the necessary hyperparameters and configurations for running experiments. Examples and templates for these YAML files can be found within each submodule.

- **Nyuntam Text Generation**:
  - Refer to [dataset imports](https://nyunai.github.io/nyuntam-docs/dataset/) and [models imports](https://nyunai.github.io/nyuntam-docs/model/) for configuration details.
  - Scripts and example YAML files are available [here](https://github.com/nyunAI/nyuntam-text-generation/tree/main/scripts).

- **Nyuntam Vision**:
  - Example YAML configurations are can be found [here](https://github.com/nyunAI/nyuntam-vision/tree/main/scripts). 
  - Details on setting up model & dataset paths, compression algorithms, and other parameters can be found in the [documentation here](https://nyunai.github.io/nyuntam-docs/nyuntam_vision/).

- **Nyuntam Adapt**:
  - For fine-tuning and transfer learning tasks, example YAML configurations are available [here](https://github.com/nyunAI/nyuntam_adapt/tree/main/scripts)
  - Detailed explanations on model & dataset path setups, params, and algorithms are available in the [documentation](https://nyunai.github.io/nyuntam-docs/adapt/).

### Running Experiments

To run an experiment, use the following command from within the appropriate submodule directory:

```bash
python main.py --yaml_path {path/to/recipe.yaml}
```

This command will execute the main script using the configurations specified in the YAML file.

## Examples

For detailed examples and use cases, refer to [examples](./examples/readme.md)

## Documentation

For complete details checkout [NyunAI Docs](https://nyunai.github.io/nyuntam-docs)
