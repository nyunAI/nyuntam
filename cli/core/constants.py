from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

from pathlib import Path
from typing import Union, Dict, Optional


# ==============================================================
#                       Logging Messages
# ==============================================================


# Workspace messages
class WorkspaceMessage(StrEnum):

    # error
    WORKSPACE_PATH_DOES_NOT_EXIST = "Workspace path does not exist: {}"

    # info
    WORKSPACE_SPEC_FOUND = "Workspace spec found."
    CREATED_NEW_WORKSPACE = "Creating a new workspace spec."

    CUSTOM_DATA_PATH_ALREADY_EXISTS = "A custom data path already exists. Existing path: {current_path}. Given path: {given_path}."
    CUSTOM_DATA_PATH_UPDATED = (
        "Custom data path updated, from: {from_path} to: {to_path} in workspace spec."
    )

    EXTENSION_ALREADY_EXISTS = "A different extension set already exists. Existing extension(s): {current_extensions}. Given extension(s): {given_extensions}."
    EXTENSION_UPDATED = "Extension(s) updated, from: {from_extensions} to {to_extensions} in workspace spec."

    WORKSPACE_INITIALIZED = "Workspace initialized."


# Workspace extension
class WorkspaceExtension(StrEnum):

    TEXT_GENERATION = "nyuntam-text-generation"

    ALL = "all"
    NONE = "none"

    @staticmethod
    def get_extensions_list(
        extension: Union["WorkspaceExtension", Dict["WorkspaceExtension", str]]
    ):
        if isinstance(extension, Dict):
            extension = {
                WorkspaceExtension(key): value for key, value in extension.items()
            }
            if all(val == "True" for val in extension.values()):
                return [WorkspaceExtension.ALL]

            retlist = [ext for ext, value in extension.items() if value == "True"]
            return retlist
        elif extension == WorkspaceExtension.ALL:
            return [
                WorkspaceExtension.TEXT_GENERATION,
            ]
        else:
            return [extension]

    @staticmethod
    def get_extensions_dict(
        *extension: "WorkspaceExtension",
    ) -> Dict["WorkspaceExtension", str]:
        if any(WorkspaceExtension(ext) == WorkspaceExtension.ALL for ext in extension):
            return {
                WorkspaceExtension.TEXT_GENERATION: "True",
            }
        elif any(
            WorkspaceExtension(ext) == WorkspaceExtension.NONE for ext in extension
        ):
            return {
                WorkspaceExtension.TEXT_GENERATION: "False",
            }
        else:
            # iteratively carry ahead the dictionary and update the extension
            _extensions = WorkspaceExtension.get_extensions_dict(
                WorkspaceExtension.NONE
            )
            for ext in extension:
                _extensions.update({WorkspaceExtension(ext): "True"})
            return _extensions

    @staticmethod
    def is_extension_different(
        extension: Dict["WorkspaceExtension", str],
        other_extension: Dict["WorkspaceExtension", str],
    ):

        # cast keys to WorkspaceExtension
        extension = {WorkspaceExtension(key): value for key, value in extension.items()}
        other_extension = {
            WorkspaceExtension(key): value for key, value in other_extension.items()
        }
        return extension != other_extension


# Workspace spec
class WorkspaceSpec(StrEnum):
    # key constants
    WORKSPACE = "Workspace"
    CUSTOM_DATA = "CustomData"
    LOGS = "Logs"
    EXTENSIONS = "Extensions"

    PATH = "path"

    # path constants
    NYUN = ".nyunservices"
    WORKSPACE_SPEC = "workspace.spec"
    LOG_FILE = "zero.log"
    ENV = ".env"

    @staticmethod
    def get_workspace_spec_path(workspace_path: Path):
        return workspace_path / WorkspaceSpec.NYUN / WorkspaceSpec.WORKSPACE_SPEC

    @staticmethod
    def get_workspace_spec_dir(workspace_path: Path):
        return workspace_path / WorkspaceSpec.NYUN

    @staticmethod
    def get_log_file_path(workspace_path: Path):
        return workspace_path / WorkspaceSpec.NYUN / WorkspaceSpec.LOG_FILE

    @staticmethod
    def get_env_file_path(workspace_path: Optional[Path]):
        env_path = workspace_path / WorkspaceSpec.ENV
        return env_path if env_path.exists() else None


# ==============================================================
#                       Docker Constants
# ==============================================================


class DockerRepository(StrEnum):
    NYUNTAM_TEXT_GENERATION = "nyunadmin/nyuntam-text-generation"
    NYUNTAM_TEXT_GENERATION_TENSORRT_LLM = (
        "nyunadmin/nyuntam-text-generation-tensorrt-llm"
    )


class DockerTag(StrEnum):
    # text generation
    FLAP = "flap"
    TENSORRTLLM =    "tensorrtllm"
    AWQ = "autoawq"

    LATEST = "latest"
    # public v0.1
    V0_1 = "v0.1"
    PUBLIC_LATEST = V0_1


class Platform(StrEnum):
    HUGGINGFACE = "huggingface"
    TIMM = "timm"



class Algorithm(StrEnum):
    # text-generation
    AUTOAWQ = "AutoAWQ"
    TENSORRTLLM = "TensorRTLLM"
    FLAPPRUNER = "FlapPruner"
    TENSORRT = "TensorRT"

    TEXT_GENERATION = "text_generation"


# yaml keys
class YamlKeys(StrEnum):
    ALGORITHM = "ALGORITHM"
    PLATFORM = "PLATFORM"

    # adapt
    TASK = "TASK"


class DockerPath(Enum):
    SCRIPT = Path("/scripts")
    WORKSPACE = Path("/workspace")
    USER_DATA = Path("/user_data")
    NYUNTAM = Path("/nyuntam")
    CUSTOM_DATA = Path("/custom_data")

    @staticmethod
    def get_customdata_path_in_docker(
        workspace_extension: Union[WorkspaceExtension, str]
    ) -> Path:
        if WorkspaceExtension(workspace_extension) == WorkspaceExtension.TEXT_GENERATION:
            return DockerPath.WORKSPACE.value / "custom_data"

    @staticmethod
    def get_script_path_in_docker(script_path: Path):
        return DockerPath.SCRIPT.value / script_path.name

    @staticmethod
    def get_nyuntam_path_in_docker():
        return DockerPath.NYUNTAM.value


class DockerCommand(StrEnum):
    # Base commands - removed clone commands, using mounted directory
    RUN = "python /nyuntam/main.py --yaml_path {script_path}"

    @staticmethod
    def get_run_command(script_path: Union[Path, str], algorithm: Optional[Algorithm] = None) -> str:
        """
        Get the appropriate run command based on the algorithm.
        Uses mounted nyuntam directory instead of cloning.
        """
        return f"/bin/bash -c '{DockerCommand.RUN.format(script_path=script_path)}'"


NYUN_ENV_KEY_PREFIX = "NYUN_"
EMPTY_STRING = ""
