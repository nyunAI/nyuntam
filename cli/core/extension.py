from cli.core.constants import (
    WorkspaceExtension,
    DockerRepository,
    DockerTag,
    Algorithm,
    Platform,
    YamlKeys,
)
from cli.core.utils import pull_docker_image
from cli.core.models import NyunDocker
from typing import Any, Set, List, Dict, Union, Tuple, Optional
from pathlib import Path
import logging
from docker.models.containers import ExecResult, Container

logger = logging.getLogger(__name__)


class DockerMetadata:
    # a class to store extension metadata: Algorithm, NyunDocker, BaseExtension

    def __init__(
        self,
        platforms: List[Platform],
        algorithm: Algorithm,
        docker_image: NyunDocker,
        extension: WorkspaceExtension,
    ):
        self.platforms = platforms
        self.algorithm = algorithm
        self.docker_image = docker_image
        self.extension_type = extension


class BaseExtension:

    extension_type: Union[WorkspaceExtension, None] = None
    docker_images: Set[NyunDocker] = set()
    extension_metadata: Set[DockerMetadata] = set()

    _all_docker_images: Set[NyunDocker] = set()
    _registry: Set[DockerMetadata] = set()

    def __init__(self):
        self.installed = False
        for image in self.docker_images:
            self._all_docker_images.add(image)

        for meta in self.extension_metadata:
            self.register(meta)

    def register(self, metadata: DockerMetadata):
        self._registry.add(metadata)

    def filter_registry(
        self,
        algorithm: Union[None, Algorithm] = None,
        platform: Union[None, Platform] = None,
    ) -> List[DockerMetadata]:

        if not algorithm and not platform:
            return self._registry
        if not algorithm:
            return list(filter(lambda meta: platform in meta.platforms, self._registry))
        if not platform:
            return list(
                filter(lambda meta: algorithm == meta.algorithm, self._registry)
            )
        return list(
            filter(
                lambda meta: algorithm == meta.algorithm and platform in meta.platforms,
                self._registry,
            )
        )

    def install(self):
        if len(self._all_docker_images) == 0:
            raise ValueError(f"No docker images found for {self.extension_type}")

        # parallel pull
        pull_docker_image(*self._all_docker_images)

        # or sequencially do img.install() for each image in self._all_docker_images

    def uninstall(self):
        print(f"Uninstalling {self.extension_type}")
        # TODO
        # call utils.uninstall
        self.installed = False

    def run(self, file_path: Path, workspace: "Workspace", log_path: Optional[Path] = None) -> Container:
        # find from registry the metadata that has algorithm and then find the corresponding NyunDocker; then for the NyunDocker trigger the .run()
        import yaml

        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        try:
            if not data.get(YamlKeys.ALGORITHM, data.get(YamlKeys.TASK, False)):
                raise KeyError("Atleast one of 'ALGORITHM' or 'TASK' key is required.")

            algorithm = Algorithm(
                data.get(YamlKeys.ALGORITHM) or data.get(YamlKeys.TASK)
            )
            platform = (
                Platform(data.get(YamlKeys.PLATFORM))
                if data.get(YamlKeys.PLATFORM)
                else None
            )
        except Exception as e:
            logger.error(e)
            raise Exception from e

        metadata = self.filter_registry(algorithm=algorithm, platform=platform)
        metadata = metadata[0] if len(metadata) else None

        print("Extension type:", metadata.extension_type)
        print("Algorithm:", metadata.algorithm)
        print("Platforms:", [str(platform) for platform in metadata.platforms])

        if metadata is None:
            raise ValueError(f"No docker image found for algorithm: {algorithm}")
        return metadata.docker_image.run(file_path, workspace, metadata, log_path=log_path)


class KompressVisionExtension(BaseExtension):

    # => kompress-vision

    extension_type = WorkspaceExtension.VISION
    docker_images = {
        NyunDocker(DockerRepository.NYUN_KOMPRESS, DockerTag.KOMPRESS_MMRAZOR),
        NyunDocker(DockerRepository.NYUN_ZERO_VISION, DockerTag.PUBLIC_LATEST),
    }
    extension_metadata = {
        DockerMetadata(
            algorithm=Algorithm.FXQUANT,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_VISION, DockerTag.PUBLIC_LATEST
            ),
            platforms=[Platform.TORCHVISION],
            extension=WorkspaceExtension.VISION,
        ),
        DockerMetadata(
            algorithm=Algorithm.KDTRANSFER,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_VISION, DockerTag.PUBLIC_LATEST
            ),
            platforms=[Platform.TIMM],
            extension=WorkspaceExtension.VISION,
        ),
        DockerMetadata(
            algorithm=Algorithm.MMRAZOR,
            docker_image=NyunDocker(
                DockerRepository.NYUN_KOMPRESS, DockerTag.KOMPRESS_MMRAZOR
            ),
            platforms=[Platform.MMYOLO, Platform.MMDET],
            extension=WorkspaceExtension.VISION,
        ),
        DockerMetadata(
            algorithm=Algorithm.MMRAZORDISTILL,
            docker_image=NyunDocker(
                DockerRepository.NYUN_KOMPRESS, DockerTag.KOMPRESS_MMRAZOR
            ),
            platforms=[
                Platform.MMSEG,
                Platform.MMPOSE,
                Platform.MMDET,
                Platform.MMYOLO,
            ],
            extension=WorkspaceExtension.VISION,
        ),
        DockerMetadata(
            algorithm=Algorithm.NNCF,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_VISION, DockerTag.KOMPRESS_MMRAZOR
            ),
            platforms=[
                Platform.MMDET,
            ],
            extension=WorkspaceExtension.VISION,
        ),
        DockerMetadata(
            algorithm=Algorithm.NNCF,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_VISION, DockerTag.PUBLIC_LATEST
            ),
            platforms=[
                Platform.TORCHVISION,
            ],
            extension=WorkspaceExtension.VISION,
        ),
        DockerMetadata(
            algorithm=Algorithm.NNCFQAT,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_VISION, DockerTag.PUBLIC_LATEST
            ),
            platforms=[
                Platform.TIMM,
            ],
            extension=WorkspaceExtension.VISION,
        ),
        DockerMetadata(
            algorithm=Algorithm.ONNXQUANT,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_VISION, DockerTag.PUBLIC_LATEST
            ),
            platforms=[
                Platform.MMDET,
                Platform.MMYOLO,
                Platform.MMSEG,
                Platform.MMPOSE,
                Platform.TIMM,
            ],
            extension=WorkspaceExtension.VISION,
        ),
        DockerMetadata(
            algorithm=Algorithm.TENSORRT,
            docker_image=NyunDocker(
                DockerRepository.NYUN_KOMPRESS, DockerTag.PUBLIC_LATEST
            ),
            platforms=[
                Platform.MMDET,
                Platform.MMYOLO,
                Platform.MMSEG,
                Platform.MMPOSE,
                Platform.TORCHVISION,
            ],
            extension=WorkspaceExtension.VISION,
        ),
        DockerMetadata(
            algorithm=Algorithm.TORCHPRUNE,
            docker_image=NyunDocker(
                DockerRepository.NYUN_KOMPRESS, DockerTag.PUBLIC_LATEST
            ),
            platforms=[Platform.TIMM, Platform.TORCHVISION],
            extension=WorkspaceExtension.VISION,
        ),
    }


class KompressTextGenerationExtension(BaseExtension):
    extension_type = WorkspaceExtension.TEXT_GENERATION
    docker_images = {
        # NyunDocker(DockerRepository.NYUN_KOMPRESS, DockerTag.MLCLLM),
        # NyunDocker(DockerRepository.NYUN_KOMPRESS, DockerTag.EXLLAMA),
        # public
        NyunDocker(DockerRepository.NYUN_ZERO_TEXT_GENERATION, DockerTag.LATEST),
        NyunDocker(
            DockerRepository.NYUN_ZERO_TEXT_GENERATION_TENSORRT_LLM,
            DockerTag.PUBLIC_LATEST,
        ),
 
    }

    extension_metadata = {
        # DockerMetadata(
        #     algorithm=Algorithm.EXLLAMA,
        #     docker_image=NyunDocker(DockerRepository.NYUN_KOMPRESS, DockerTag.EXLLAMA),
        #     platforms=[Platform.HUGGINGFACE],
        #     extension=WorkspaceExtension.TEXT_GENERATION,
        # ),
        # DockerMetadata(
        #     algorithm=Algorithm.MLCLLM,
        #     docker_image=NyunDocker(DockerRepository.NYUN_KOMPRESS, DockerTag.MLCLLM),
        #     platforms=[Platform.HUGGINGFACE],
        #     extension=WorkspaceExtension.TEXT_GENERATION,
        # ),
        # public
        DockerMetadata(
            algorithm=Algorithm.AUTOAWQ,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_TEXT_GENERATION, DockerTag.LATEST
            ),
            platforms=[Platform.HUGGINGFACE],
            extension=WorkspaceExtension.TEXT_GENERATION,
        ),
        DockerMetadata(
            algorithm=Algorithm.TENSORRTLLM,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_TEXT_GENERATION_TENSORRT_LLM,
                DockerTag.PUBLIC_LATEST,
            ),
            platforms=[Platform.HUGGINGFACE],
            extension=WorkspaceExtension.TEXT_GENERATION,
        ),
        DockerMetadata(
            algorithm=Algorithm.FLAPPRUNER,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_TEXT_GENERATION, DockerTag.LATEST
            ),
            platforms=[Platform.HUGGINGFACE],
            extension=WorkspaceExtension.TEXT_GENERATION,
        ),
    }


class AdaptExtension(BaseExtension):
    extension_type = WorkspaceExtension.ADAPT
    docker_images = {
        # public
        NyunDocker(DockerRepository.NYUN_ZERO_ADAPT, DockerTag.PUBLIC_LATEST),
        NyunDocker(DockerRepository.NYUN_ADAPT, DockerTag.ADAPT),
    }
    extension_metadata = {
        # huggingface - 'text_classification'
        DockerMetadata(
            algorithm=Algorithm.TEXT_CLASSIFICATION,
            docker_image=NyunDocker(DockerRepository.NYUN_ADAPT, DockerTag.ADAPT),
            platforms=[Platform.HUGGINGFACE],
            extension=WorkspaceExtension.ADAPT,
        ),
        # mmdet - 'detection'
        DockerMetadata(
            algorithm=Algorithm.DETECTION,
            docker_image=NyunDocker(DockerRepository.NYUN_ADAPT, DockerTag.ADAPT),
            platforms=[Platform.MMDET],
            extension=WorkspaceExtension.ADAPT,
        ),
        # mmpose - pose est
        DockerMetadata(
            algorithm=Algorithm.POSE_DETECTION,
            docker_image=NyunDocker(DockerRepository.NYUN_ADAPT, DockerTag.ADAPT),
            platforms=[Platform.MMPOSE],
            extension=WorkspaceExtension.ADAPT,
        ),
        # mmseg - segmentation
        DockerMetadata(
            algorithm=Algorithm.SEGMENTATION,
            docker_image=NyunDocker(DockerRepository.NYUN_ADAPT, DockerTag.ADAPT),
            platforms=[Platform.MMSEG],
            extension=WorkspaceExtension.ADAPT,
        ),
        ## public
        # huggingface - 'text_generation'
        DockerMetadata(
            algorithm=Algorithm.TEXT_GENERATION,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_ADAPT, DockerTag.PUBLIC_LATEST
            ),
            platforms=[Platform.HUGGINGFACE],
            extension=WorkspaceExtension.ADAPT,
        ),
        # huggingface - 'question_answering
        DockerMetadata(
            algorithm=Algorithm.QUESTION_ANSWERING,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_ADAPT, DockerTag.PUBLIC_LATEST
            ),
            platforms=[Platform.HUGGINGFACE],
            extension=WorkspaceExtension.ADAPT,
        ),
        # huggingface - 'Seq2Seq_tasks.summarization', 'Seq2Seq_tasks.translation'
        DockerMetadata(
            algorithm=Algorithm.SEQ2SEQ_TASKS,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_ADAPT, DockerTag.PUBLIC_LATEST
            ),
            platforms=[Platform.HUGGINGFACE],
            extension=WorkspaceExtension.ADAPT,
        ),
        # hf/timm - image classification
        DockerMetadata(
            algorithm=Algorithm.IMAGE_CLASSIFICATION,
            docker_image=NyunDocker(
                DockerRepository.NYUN_ZERO_ADAPT, DockerTag.PUBLIC_LATEST
            ),
            platforms=[Platform.HUGGINGFACE, Platform.TIMM],
            extension=WorkspaceExtension.ADAPT,
        ),
    }
