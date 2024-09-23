
### 8-bit CPU Quantization of ResNet50 using NNCF on CIFAR-10 Dataset via NyunCLI Installation

8-bit quantization of Image Classification models are done via NNCF, ONNX Quantizers for CPU and TensorRT Quantizer for GPU deployment. This example shows how to Quantize a ResNet50 model with NNCF for CPU Quantization via nyuncli 
#### Table of Contents 
1. [Downloading NyunCLI](#downloading-nyuncli) 
2. [Downloading the Dataset](#downloading-the-dataset) 
3. [Model Loading](#model-loading) 
4. [Loading Pretrained Weights (Optional)](#loading-pretrained-weights-optional)
5. [Finetuning Model Before Training (Optional)](#finetuning-model-before-training-optional) 
6. [Starting the Job](#starting-the-job) 
7. [Results](#results)
#### 1. Downloading NyunCLI
Nyun-cli offers the users the luxury to compress their neural networks via a single command line. To download and install NyunCLI.
```bash
pip install git+https://github.com/nyunAI/nyunzero-cli.git
nyun init {WORKSPACE_PATH} ""
```
Here ``WORKSPACE_PATH`` is the root folder for running the experimentation and ``CUSTOM_DATA_PATH`` defines the path of custom data. The ``YAML_PATH`` is the path of configuration to be used. The base configuration used for this example can be found at [vision scripts]().This example offers a basic example for  installing nyuncli visit nyuncli documentation for a more advanced installation.
#### 2. Downloading the dataset
Downloading and formatting CIFAR-10 dataset is automated via Nyuntam. We need to specify the following hyperparameters in the YAML configuration 
```yaml
DATASET_NAME: CIFAR10
DATASET_PATH: {DATASET_PATH}
TASK: image_classification
``` 
Note: Dataset Path is expected to to be the path relative to WORKSPACE_PATH
#### 3. Model Loading
Nyuntam supports most Classification models supported via HuggingFace, Timm and Torchvision libraries. We load ResNet50 from torchvision in this experiment. The following parameters in the YAML configuration are to be updated
```yaml
MODEL_NAME: resnet50
PLATFORM: torchvision
```
#### 4. Loading Pretrained Weights (Optional)
You can optionally load pretrained weights if you already have the same. In case of this tutorial we would be finetuning the CIFAR 10 on 10 epochs instead (see sec.5)
```yaml
CUSTOM_MODEL_PATH: {CUSTOM_MODEL_PATH}
#leave CUSTOM_MODEL_PATH as "" if unused
```
#### 5. Finetuning Model Before Training (Optional)
You can optionally fine-tune the model with these parameters to tune in with the custom dataset. 
```yaml
TRAINING: True
LEARNING_RATE: 0.001
FINETUNE_EPOCHS: 10
VALIDATE: True
VALIDATION_INTERVAL: 1
```
#### 6. Starting the Job
The following command starts the job using nyun-cli. The NNCF 
```bash
nyun run {CONFIG_PATH}
```
``CONFIG Path`` Config path is the edited yaml used for defining the hyperparameters of the job. 

After the job is completed the folder structure will be as follows
```
{WORKSPACE}
├── datasets
│   ├── {JOB_ID}
│   |	├── train
│   |	├── val
├── logs
│   ├── log.log
├── jobs
│   ├── mds.xml
│   ├── mds.bin
```
	
####  7. Results

The model post finetuning on cifar-10 for 10 epochs had validation 67.4 % validation accuracy and after quantization had an accuracy of 67.2 %. To benchmark the reduced latency of the model use [NNCF Benchmarking tool](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html) using the following command. 

```bash
benchmark_app -m {path to .xml file} -d CPU -api async
```
The Final Results are as follows:

| Model | Accuracy | Latency |
|-------|----------|---------|
| FP32  | 67.7     | 1.32 ms |
| INT8  | 67.2     | 0.68 ms |

