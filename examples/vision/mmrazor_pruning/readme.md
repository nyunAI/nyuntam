
### MMRazor Pruning on YOLOX using NyunCLI

Pruning Object Detection is supported via MMRazor Pruning. Currently activation based pruning and flop based pruning are the available option. This example utilizes NyunCLI a 1-click solution to run the pruning job. We need to build a configuration for the job specifying parameters. The steps for pruning a YOLOX is as follows. Starting YAML [Here](https://github.com/nyunAI/nyuntam-vision/blob/Scripts/scripts/prune/object_detection/MMRazorPruneMMDet.yaml)
#### Table of Contents 
1. [Downloading NyunCLI](#downloading-nyuncli) 
2. [Downloading the Dataset](#downloading-the-dataset) 
3. [Model Loading](#model-loading) 
4. [Loading Pretrained Weights (Optional)](#loading-pretrained-weights-optional)
5. [Modifying Hyperparameters](#setting-task-specific-hyperparameters) 
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
DATASET_NAME: COCODETECTION
DATASET_PATH: {DATASET_PATH}
TASK: object_detection
``` 
Note: Dataset Path is expected to to be the path relative to WORKSPACE_PATH
#### 3. Model Loading
Nyuntam supports object detection models from MMDet and MMYolo libraries, we load YOLOX-tiny from MMDet for this experiment. Pretrained COCO weights and model config are internally downloaded during job run. NOTE: the model name argument must match the exact name used in mmdet configs. 
```yaml
MODEL_NAME: yolox_tiny_8xb8-300e_coco
PLATFORM: mmdet
```
#### 4. Loading Pretrained Weights (Optional)
You can optionally load pretrained weights if you already have the same.  This can be used to prune mmdet and mmyolo models trained on custom datasets. 
```yaml
CUSTOM_MODEL_PATH: {CUSTOM_MODEL_PATH}
#leave CUSTOM_MODEL_PATH as "" if unused
```
#### 5. Setting Task Specific Hyperparameters
We specify the hyperparameters used for running this tutorial below, we may modify them to better suit your pruning task, more details on the hyperparameters can be found at [Nyun Documentation](https://nyunai.github.io/nyun-docs/kompress/algorithms/).
```yaml
    INTERVAL: 10 # The interval between two pruning operations
    NORM_TYPE: 'act' # Type of importance , activation or flops
    LR_RATIO: 0.1 # Learning Rate Ratio
    TARGET_FLOP_RATIO: 0.9 # Target Number of Flops value * number of existing flops.
    EPOCHS: 5 # Total Number of Epochs to perform pruning (stops early once reached required flops)
```
#### 6. Starting the Job
The following command starts the job using nyun-cli. 
```bash
nyun run {CONFIG_PATH}
```
``CONFIG Path`` Config path is the edited yaml used for defining the hyperparameters of the job. 

After the job is completed the folder structure at Workspace_path will be as follows
```
{WORKSPACE}
├── datasets
│   ├── {JOB_ID}
|	|	├──root
│   |	|	├── train
│   |	|	├── val
│   |	|	├── annotations
├── logs
│   ├── log.log
├── jobs
│   ├── mds.pt
│   ├── flops_{target_flop_ratio}.pth
|	├── fix_subnet.json
```
	
####  7. Results

We use MAP and Latency to assess the result produced. 
To calculate the latency and map post pruning you can use mmrazor's test.py using the following code
```bash
cd {mmrazor tools folder}
python test.py {cache_path}/current_fisher_finetune_config.py.py {path to mds.pt} {batch_size}
```

The results are as follows:

| Model          | MAP  | Latency |
|----------------|------|---------|
| Unpruned       | 32.0 | 0.35    |
| Prune+Finetune | 29.6 | 0.31 ms |
