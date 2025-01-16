# Nyuntam Text Generation
Nyuntam Text Generation contains the sota compression methods and algorithms to achieve efficiency on text-generation tasks (primarily operated on large language models).
This module implements model efficiency mixins via 
- pruning
- quantization
- accelerations with tensorrtllm


## Installation
Installation can be performed either via installing requirements in a virtual environment or by utilizing our docker images. To quickly run Kompress for experimentation and usage, utilize Nyun CLI to get Kompress running in no time. For contributing to Nyuntam build docker containers from the available docker image or create a virtual enviroment from the provided requirements.txt.

### Nyunzero CLI
The recommended method to install and use nyuntam is via the nyunzero-cli. Further details can be found here : [NyunZero CLI](https://github.com/nyunAI/nyunzero-cli)

### Git + Docker
Nyuntam (Kompress) can also be used by cloning the repository and pulling hhe required docker. 

1. **Git Clone** : First, clone the repository to your local machine:
    ```bash
    $ git clone --recursive https://github.com/nyunAI/nyuntam.git
    $ cd nyuntam
    ```

2. **Docker Pull**: Next, pull the corresponding docker container(s) and run it :

[(list of nyunzero dockers)](https://hub.docker.com/?search=nyunzero)

    ```bash 
    $ docker pull <docker>

    $ docker run -it -d --gpus all -v $(pwd):/workspace <docker_image_name_or_id> bash 
    ```

<span style="color:red">**NOTE:**</span> 
- nvidia-container-toolkit is expected to be installed before the execution of this
- all docker mount tags and environment tags holds (add hf tokens for access to gated repos within the dockers)


### Git + virtual environment

Nyuntam can also be used by cloning the repository and setting up a virtual environment. 

1. **Git Clone** : First, clone the repository to your local machine:
    ```bash
    $ git clone --recursive https://github.com/nyunAI/nyuntam.git
    $ cd nyuntam
    ```

2. **Create a virtual environment using Venv**
   ```sh
   python3 -m venv {ENVIRONMENT_NAME}
   source {ENVIRONMENT_NAME}/bin/activate
   ```

3. **Pip install requirements**
   ```sh
   pip install -r nyuntam-text-generation/requirements.txt
   ```

   **note:** for tensorrtllm, we recommend using the dockers directly.

## Usage 

### Setting up the YAML files
If the dataset and models weights exist online (huggingface hub) then the experiments can be started withing no time. Kompress requires a recipes with all the required hyperparameters and arguments to compress a model.
find a set of [compression scripts here](https://github.com/nyunAI/nyuntam-text-generation/tree/main/scripts)

The major hyperparameters are metioned below : 

- ***Dataset***
```yaml
DATASET_NAME: "wikitext"
DATASET_SUBNAME: "wikitext-2-raw-v1"
DATA_PATH: ""  # custom data path (loadable via datasets.load_from_disk)
TEXT_COLUMN: "text" # if multiple, separate by comma e.g. 'instruction,input,output'
SPLIT: "train"
FORMAT_STRING: # format string for multicolumned datasets

```

- ***Model***

```yaml
MODEL: Llama-3
MODEL: "meta-llama/Meta-Llama-3-8B"  # hf repo id's
CUSTOM_MODEL_PATH: ""
```

for details on dataset and model configurations checkout [nyuntam-docs/nyuntam_text_generation](https://nyunai.github.io/nyuntam-docs/nyuntam_text_generation/)

### Run command
```sh
python main.py --yaml_path {path/to/recipe}
```

This command runs the main file with the configuration setup in the recipe.


## Acknowledgments and Citations

This repository utilizes various state-of-the-art methods and algorithms developed by the research community. We acknowledge the following works that have contributed to the development and performance of the Nyuntam Text Generation module:

- [**Extreme Compression of Large Language Models via Additive Quantization**](https://arxiv.org/abs/2401.06118)  
  *Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh.* arXiv preprint, 2024.
  ```bibtex
  @misc{egiazarian2024extreme,
      title={Extreme Compression of Large Language Models via Additive Quantization},
      author={Vage Egiazarian and Andrei Panferov and Denis Kuznedelev and Elias Frantar and Artem Babenko and Dan Alistarh},
      year={2024},
      eprint={2401.06118},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
  }
  ```
- [**PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression**](https://arxiv.org/abs/2405.14852)  
  *Vladimir Malinovskii, Denis Mazur, Ivan Ilin, Denis Kuznedelev, Konstantin Burlachenko, Kai Yi, Dan Alistarh, Peter Richtarik.* arXiv preprint, 2024.
  ```bibtex
  @misc{malinovskii2024pvtuning,
      title={PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression},
      author={Vladimir Malinovskii and Denis Mazur and Ivan Ilin and Denis Kuznedelev and Konstantin Burlachenko and Kai Yi and Dan Alistarh and Peter Richtarik},
      year={2024},
      eprint={2405.14852},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
  }
  ```

- [**QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving**](https://arxiv.org/abs/2405.04532)  
  *Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han.* arXiv preprint, 2024.
  ```bibtex
  @article{lin2024qserve,
      title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
      author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
      journal={arXiv preprint arXiv:2405.04532},
      year={2024}
  }
  ```

- [**Fluctuation-based Adaptive Structured Pruning for Large Language Models**](https://arxiv.org/abs/2312.11983)  
  *Yongqi An, Xu Zhao, Tao Yu, Ming Tang, Jinqiao Wang.* arXiv preprint, 2023.
  ```bibtex
  @misc{an2023fluctuationbased,
      title={Fluctuation-based Adaptive Structured Pruning for Large Language Models},
      author={Yongqi An and Xu Zhao and Tao Yu and Ming Tang and Jinqiao Wang},
      year={2023},
      eprint={2312.11983},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
  }
  ```

- [**AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**](https://arxiv.org/abs/2306.00978)  
  *Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Song Han.* arXiv, 2023.
  ```
  @article{lin2023awq,
      title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
      author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
      journal={arXiv},
      year={2023}
  }
  ```

- [**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM)  
  *NVIDIA.*
