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


class NyuntamTextGenerationExtension(BaseExtension):
    extension_type = WorkspaceExtension.TEXT_GENERATION
    docker_images = {
        NyunDocker(DockerRepository.NYUNTAM_TEXT_GENERATION, DockerTag.LATEST),
        NyunDocker(
            DockerRepository.NYUNTAM_TEXT_GENERATION_TENSORRT_LLM,
            DockerTag.LATEST,
        ),
    }

    extension_metadata = {
        DockerMetadata(
            algorithm=Algorithm.AUTOAWQ,
            docker_image=NyunDocker(
                DockerRepository.NYUNTAM_TEXT_GENERATION,
                DockerTag.LATEST,
            ),
            platforms=[Platform.HUGGINGFACE],
            extension=WorkspaceExtension.TEXT_GENERATION,
        ),
        DockerMetadata(
            algorithm=Algorithm.TENSORRTLLM,
            docker_image=NyunDocker(
                DockerRepository.NYUNTAM_TEXT_GENERATION_TENSORRT_LLM,
                DockerTag.LATEST,
            ),
            platforms=[Platform.HUGGINGFACE],
            extension=WorkspaceExtension.TEXT_GENERATION,
        ),
        DockerMetadata(
            algorithm=Algorithm.FLAPPRUNER,
            docker_image=NyunDocker(
                DockerRepository.NYUNTAM_TEXT_GENERATION,
                DockerTag.LATEST,
            ),
            platforms=[Platform.HUGGINGFACE],
            extension=WorkspaceExtension.TEXT_GENERATION,
        ),
    }
