# Nyuntam 🚀

**Nyuntam** is NyunAI's cutting-edge toolkit for optimizing and accelerating large language models (LLMs) through state-of-the-art compression techniques. 🛠️ With an integrated CLI, managing your workflows and experimenting with various compression methods has never been easier! ✨

## Quick Start ⚡

Ready to dive in? Here's a minimal example to get you up and running with Nyuntam:

1.  **Initialize Your Workspace:** 🗂️
    First, set up your workspace using the `nyun init` command. This creates the necessary directories and configurations for your experiments.

    ```bash
    nyun init ~/my-workspace ~/my-data --extensions text-gen
    ```

    This command initializes a workspace at `~/my-workspace`, sets the custom data path to `~/my-data`, and installs the `text-gen` extension.

2.  **Run an Example Experiment:** 🏃‍♀️
    Now, run an example experiment using a pre-configured YAML file. For instance, to try out FLAP pruning:

    ```bash
    nyun run examples/text-generation/flap_pruning/config.yaml
    ```

    This command executes the main script using the configurations specified in the provided YAML file.

## Key Features ✨

-   **State-of-the-Art Compression**: 🗜️ Includes advanced techniques like pruning, quantization, and distillation to ensure model efficiency without sacrificing performance.
-   **Multi-Platform Support**: 💻 Run experiments seamlessly on various platforms using Docker or virtual environments.
-   **Integrated CLI**: ⌨️ Built-in command-line interface (`nyun`) for easy workspace management and experiment execution.
-   **Extensible Architecture**: 🧩 Supports various SOTA compression algorithms, using a single cli command.
## Installation 🛠️

### Prerequisites

-   Python 3.8 or later
-   For GPU support: NVIDIA Container Toolkit (when using Docker) 🐳

### Quick Install

Install Nyuntam using pip:

```bash
pip install nyuntam
```

### Alternative Installation Methods

#### Git + Docker 🐳

1.  **Install NVIDIA Container Toolkit** (Linux):
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

2.  **Clone and Setup**:
    ```bash
    git clone --recursive https://github.com/nyunAI/nyuntam.git
    cd nyuntam
    docker pull nyunadmin/nyuntam-text-generation:latest
    docker run -it -d --gpus all -v $(pwd):/workspace --name nyuntam-dev --network=host nyunadmin/nyuntam-text-generation:latest bash
    ```

#### Git + Virtual Environment 🧪

1.  **Clone Repository**:
    ```bash
    git clone --recursive https://github.com/nyunAI/nyuntam.git
    cd nyuntam
    ```

2.  **Setup Environment**:
    ```bash
    python3 -m venv {ENVIRONMENT_NAME}
    source {ENVIRONMENT_NAME}/bin/activate
    pip install -r requirements.txt
    ```

#### Using Setup.py

## Development 🧑‍💻

This section is for developers who want to dive deep and modify the Nyuntam codebase.

### Setting Up Your Development Environment

1.  **Clone the Repository**:
    ```bash
    git clone --recursive https://github.com/nyunAI/nyuntam.git
    cd nyuntam
    ```

2.  **Choose Your Environment**:
    -   **Docker**: Follow the instructions in the "Git + Docker" section above.
    -   **Virtual Environment**: Follow the instructions in the "Git + Virtual Environment" section above.

### Working with the Code

-   **Core Scripts**: The core scripts like `main.py`, `algorithm.py`, and `commands.py` are located in the root directory of the `nyuntam` folder.
-   **Examples**: Practical examples for different tasks are located in the `nyuntam/examples` directory. Each subdirectory includes a `README.md` for guidance and `config.yaml` files for configurations.
-   **Modules**: The main modules for text generation are located in the `nyuntam/text_generation` directory.
-   **Utilities**: Utility scripts and functions are located in the `nyuntam/utils` directory.

### Running Experiments (Development)

1.  **Prepare Configuration**:
    Create a YAML file defining your experiment parameters. Example configurations are available in the `nyuntam/examples` directory.

    -   Refer to [dataset imports](https://nyunai.github.io/nyuntam-docs/dataset/) and [models imports](https://nyunai.github.io/nyuntam-docs/model/) for configuration details.
    -   Scripts and example YAML files are available [here](https://github.com/nyunAI/nyuntam-text-generation/tree/main/scripts).

2.  **Execute**:
    ```bash
    python nyuntam/main.py --yaml_path path/to/recipe.yaml
    ```

## Usage ⚙️

### Initializing a Workspace

Before running experiments, initialize your workspace:

```bash
nyun init [WORKSPACE_PATH] [CUSTOM_DATA_PATH] [OPTIONS]
```

Options:

-   `--overwrite`, `-o`: Overwrite existing workspace
-   `--extensions`, `-e`: Specify extensions to install:
    -   `text-gen`: For text generation
    -   `all`: Install all extensions
    -   `none`: No extensions

Example:

```bash
nyun init ~/my-workspace ~/my-data --extensions text-gen
```

### Running Experiments

1.  **Prepare Configuration**:
    Create a YAML file defining your experiment parameters. Example configurations are available in the `nyuntam/examples` directory.

    -   Refer to [dataset imports](https://nyunai.github.io/nyuntam-docs/dataset/) and [models imports](https://nyunai.github.io/nyuntam-docs/model/) for configuration details.
    -   Scripts and example YAML files are available [here](https://github.com/nyunAI/nyuntam-text-generation/tree/main/scripts).

2.  **Execute**:
    ```bash
    nyun run path/to/recipe.yaml
    ```

    For chained execution:

    ```bash
    nyun run script1.yaml script2.yaml
    ```

## Examples 💡

For detailed examples and use cases, check out our [examples directory](./nyuntam/examples/readme.md), which includes:

### Text Generation

-   [Maximising math performance for extreme compressions: 2-bit Llama3-8b (w2a16)](./nyuntam/examples/text-generation/aqlm_quantization/readme.md)
-   [Efficient 4-bit Quantization (w4a16) of Llama3.1-8b](./nyuntam/examples/text-generation/awq_quantization/readme.md)
-   [Llama3.1 70B: 0.5x the cost & size](./nyuntam/examples/text-generation/flap_pruning/readme.md)
-   [Achieving Up to 2.5x TensorRTLLM Speedups](./nyuntam/examples/text-generation/lmquant_quantization/readme.md)
-   [Accelerating a 4-bit Quantised Llama Model](./nyuntam/examples/text-generation/tensorrtllm_engine/readme.md)

## Documentation 📚

For complete documentation, visit [NyunAI Docs](https://nyunai.github.io/nyuntam-docs)

## Version Information ℹ️

Check your installed version:

```bash
nyun version
```

---

<span style="color:red">**NOTE:**</span> For access to gated repositories within containers, ensure you have the necessary Hugging Face tokens configured. 🔑