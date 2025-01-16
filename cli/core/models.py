from typing import Dict, Optional
from pathlib import Path
from docker.models.containers import Container  # Import Container directly from docker
from cli.core.constants import DockerRepository, DockerTag
from cli.core.utils import pull_docker_image, run_docker_container, remove_docker_image


class NyunDocker:

    registry: Dict[str, "NyunDocker"] = {}  # A register to store unique docker images

    def __new__(cls, repository: str, tag: str):
        key = f"{repository}:{tag}"
        if key not in cls.registry:
            instance = super(NyunDocker, cls).__new__(cls)
            cls.registry[key] = instance
        return cls.registry[key]

    def __init__(self, repository: DockerRepository, tag: DockerTag):
        self.repository = repository
        self.tag = tag

    def __str__(self):
        return f"{self.repository}:{self.tag}"

    def __repr__(self):
        return self.__str__()

    def _install(self):
        pull_docker_image(self)

    def _uninstall(self):
        remove_docker_image(self)

    def __hash__(self):
        return hash((self.repository, self.tag))

    def __eq__(self, other):
        if isinstance(other, NyunDocker):
            return (self.repository, self.tag) == (other.repository, other.tag)
        return False

    def run(
        self, 
        file_path: Path, 
        workspace: "Workspace", 
        metadata: "DockerMetadata",
        log_path: Optional[Path] = None
    ) -> Container:
        return run_docker_container(file_path, workspace, metadata, self, log_path=log_path)

        # TODO: except if docker is unavailable due to some reason:
        # pull docker and run again.
        # self._intall()
        # self.run(file_path)

    def install(self):
        self._install()

    def uninstall(self):
        self._uninstall()
