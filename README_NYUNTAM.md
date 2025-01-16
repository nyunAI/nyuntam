# Nyuntam

**Nyuntam** is NyunAI's advanced suite designed to optimize and accelerate large language models through state-of-the-art compression techniques.

## Key Features

- **State-of-the-Art Compression**: Includes techniques such as pruning, quantization, and distillation to ensure model efficiency without compromising performance.
- **Multi-Platform Support**: Run experiments seamlessly on various platforms using Docker or virtual environments.
- **CLI Integration**: Simplify your workflow with the [NyunZero CLI](https://github.com/nyunAI/nyunzero-cli).

## Installation

### NyunZero CLI (Recommended)

The fastest way to install and use Nyuntam is through the NyunZero CLI. It streamlines the setup process and gets you running experiments in no time.

For more details, visit the [NyunZero CLI Documentation](https://github.com/nyunAI/nyunzero-cli).

### Git + Docker

**Prerequisites: NVIDIA Container Toolkit**

#### For Linux:
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


1. **Clone the Repository**:
    ```bash
    git clone --recursive https://github.com/nyunAI/nyuntam.git
    cd nyuntam
    ```

2. **Pull the Required Docker Image**:
    ```bash
    docker pull nyunadmin/nyunzero_text_generation:v0.1
    ```

3. **Run the Docker Container**:
    ```bash
    docker run -it -d --gpus all -v $(pwd):/workspace --name nyuntam-dev --network=host nyunadmin/nyuntam-text-generation:latest bash
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
    pip install -r requirements.txt
    ```

### Using Setup.py
```bash
python install . [extra_deps]
```

The extra dependencies allow installation of individual components:
- **Text Generation**
  - "flap": For running Flap
  - "aqlm": For running AQLM
  - "autoawq": For running AutoAWQ quantization
  - "qserve": For running QServe

## Usage

### Setting Up YAML Configuration Files

Nyuntam requires a YAML file that defines all the necessary hyperparameters and configurations for running experiments. Example YAML files can be found in the `scripts` directory.

- Refer to [dataset imports](https://nyunai.github.io/nyuntam-docs/dataset/) and [models imports](https://nyunai.github.io/nyuntam-docs/model/) for configuration details.
- Scripts and example YAML files are available [here](https://github.com/nyunAI/nyuntam-text-generation/tree/main/scripts).

### Running Experiments

To run an experiment, use the following command:

```bash
python main.py --yaml_path {path/to/recipe.yaml}
```

This command will execute the main script using the configurations specified in the YAML file.

## Examples

For detailed examples and use cases, refer to [examples](./examples/readme.md)

## Documentation

For complete details checkout [NyunAI Docs](https://nyunai.github.io/nyuntam-docs)
