from setuptools import setup, find_packages
import re
from collections import Counter


def process(requirements: list) -> list:
    def get_package_names(unique_reqirements):
        unique_reqirements_stripped = [
            requirement.strip() for requirement in unique_reqirements
        ]
        unique_requirements_names_only = [
            re.split("(==|>=|<=)", requirement)[0]
            for requirement in unique_reqirements_stripped
        ]
        return unique_requirements_names_only

    def check_clashes(unique_requirements):
        unique_packages = get_package_names(unique_requirements)
        if len(set(unique_packages)) != len(unique_requirements):
            counter = Counter(unique_packages)
            repeating_packages = [
                package for package, count in counter.items() if count > 1
            ]
            raise ValueError(
                f"The following packages have multiple version dependencies {repeating_packages}"
            )

    unique_requirements = list(set(requirements))
    check_clashes(unique_requirements)
    return unique_requirements


install_requires = [
    "strenum",
    "PyYAML",
    "torch==2.3.0",
    "gdown",
    "pathtools",
    "tqdm",
    "requests==2.28.2",
    "psutil",
    "wandb",
    "wheel"
]

visionbase = [
    "opencv-python-headless==4.8.1.78",
    "Pillow==9.4.0",
    "scipy",
    "scikit-image==0.22.0",
]
onnx = ["onnx==1.15.0", "onnxruntime==1.16.3"]
nncf = onnx + ["openvino==2023.2.0", "openvino-telemetry==2023.2.1", "nncf==2.7.0"]
tensorrt = onnx + ["tensorrt==8.5.2", "pycuda"]

torchprune = ["torch-pruning==1.3.2"]

classification_datasets = ["trailmet==0.0.1rc3", "torchvision"]
classification_modelloading = [
    "huggingface==0.0.1",
    "timm==0.9.2",
    "trailmet==0.0.1rc3",
]
classification = classification_datasets + classification_modelloading
quantization = nncf + onnx + tensorrt


adaptbase = [
    "peft==0.11.1",
    "bitsandbytes==0.43.1",
    "sentence-transformers==2.2.2",
    "sentencepiece==0.2.0",
    "transformers==4.40.1",
    "triton==2.3.0",
    "trl==0.8.6",
]
accelerate = ["accelerate==0.29.3"]

textgen_base = ["transformers==4.40.1", "datasets==2.19.0", "sentencepiece==0.2.0"]
autoawq = ["autoawq", "autoawq_kernels"]

aqlm = accelerate + [
    "safetensors==0.4.3",
]

flap = [
    "packaging",
    "flash_attn",
] + adaptbase

qserve = ["Qserve"]


dependency_links = ["https://download.pytorch.org/whl/cu121", "https://pypi.nvidia.com"]
setup(
    name="nyuntam",
    version="0.0.1",
    description="Nyuntam Setup",
    long_description="Nyuntam blah",
    packages=find_packages(),
    install_requires=install_requires,
    dependency_links=dependency_links,
    extras_require={
        # vision
        "classification": process(visionbase + classification),
        "nncf": process(visionbase + nncf),
        "tensorrt": process(visionbase + tensorrt),
        "onnx": process(visionbase + onnx),
        "torchprune": process(visionbase + torchprune),
        "classificationstack": process(visionbase + classification + quantization),
        # # adapt
        "adapt": adaptbase,
        "accelerate": accelerate,
        # # text-generation
        "text-gen": process(
            textgen_base + flap + tensorrtllm + qserve + autoawq + aqlm
        ),
        "flap": process(textgen_base + flap),
        "aqlm": process(textgen_base + aqlm),
        "autoawq": process(textgen_base + autoawq),
        "qserve": process(textgen_base + qserve),
        "adapt":process(adaptbase)
    }
)
