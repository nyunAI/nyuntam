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

    VISION = "kompress-vision"
    TEXT_GENERATION = "kompress-text-generation"
    ADAPT = "adapt"

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
                WorkspaceExtension.VISION,
                WorkspaceExtension.TEXT_GENERATION,
                WorkspaceExtension.ADAPT,
            ]
        else:
            return [extension]

    @staticmethod
    def get_extensions_dict(
        *extension: "WorkspaceExtension",
    ) -> Dict["WorkspaceExtension", str]:
        if any(WorkspaceExtension(ext) == WorkspaceExtension.ALL for ext in extension):
            return {
                WorkspaceExtension.VISION: "True",
                WorkspaceExtension.TEXT_GENERATION: "True",
                WorkspaceExtension.ADAPT: "True",
            }
        elif any(
            WorkspaceExtension(ext) == WorkspaceExtension.NONE for ext in extension
        ):
            return {
                WorkspaceExtension.VISION: "False",
                WorkspaceExtension.TEXT_GENERATION: "False",
                WorkspaceExtension.ADAPT: "False",
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
    # kompress
    NYUN_KOMPRESS = "nyunadmin/nyun_kompress"

    # adapt
    NYUN_ADAPT = "nyunadmin/adapt"

    # public
    NYUN_ZERO_VISION = "nyunadmin/nyunzero_kompress_vision"
    NYUN_ZERO_TEXT_GENERATION = "nyunadmin/nyuntam-text-generation"
    NYUN_ZERO_ADAPT = "nyunadmin/nyunzero_adapt"
    NYUN_ZERO_TEXT_GENERATION_TENSORRT_LLM = (
        "nyunadmin/nyunzero_text_generation_tensorrt_llm"
    )


class DockerTag(StrEnum):
    # kompress vision
    MAIN_KOMPRESS = "main_kompress"
    KOMPRESS_MMRAZOR = "mmrazor"

    # kompress text generation
    FLAP = "flap"
    MLCLLM = "mlcllm"
    TENSORRTLLM = "tensorrtllm"
    EXLLAMA = "exllama"
    AWQ = "autoawq"

    # adapt
    ADAPT = "february"
    LATEST = "latest"
    # public v0.1
    V0_1 = "v0.1"
    PUBLIC_LATEST = V0_1


class Platform(StrEnum):
    # Platforms: {'timm', 'huggingface', 'mmpose', 'mmdet', None, 'torchvision', 'mmyolo', 'mmseg'}
    HUGGINGFACE = "huggingface"
    TIMM = "timm"
    MMPOSE = "mmpose"
    MMDET = "mmdet"
    TORCHVISION = "torchvision"
    MMYOLO = "mmyolo"
    MMSEG = "mmseg"


class Algorithm(StrEnum):

    # kompress vision
    KDTRANSFER = "KDTransfer"
    MMRAZORDISTILL = "MMRazorDistill"
    ONNXQUANT = "ONNXQuant"
    FXQUANT = "FXQuant"
    MMRAZOR = "MMRazor"
    NNCFQAT = "NNCFQAT"
    TORCHPRUNE = "TorchPrune"
    NNCF = "NNCF"

    # kompress text-generation
    AUTOAWQ = "AutoAWQ"
    MLCLLM = "MLCLLM"
    EXLLAMA = "ExLlama"
    TENSORRTLLM = "TensorRTLLM"
    FLAPPRUNER = "FlapPruner"
    TENSORRT = "TensorRT"

    # adapt
    # NOTE: The following values are essentially "TASK" values in adapt
    # TODO: Use a constant set of keys post standardization of hyperparams across Nyun

    DETECTION = "object_detection"
    IMAGE_CLASSIFICATION = "image_classification"
    POSE_DETECTION = "pose_estimation"
    QUESTION_ANSWERING = "question_answering"
    SEGMENTATION = "image_segmentation"
    SEQ2SEQ_TASKS = "Seq2Seq_tasks"
    TEXT_CLASSIFICATION = "text_classification"
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
        if WorkspaceExtension(workspace_extension) in {
            WorkspaceExtension.TEXT_GENERATION,
            WorkspaceExtension.VISION,
        }:
            return DockerPath.WORKSPACE.value / "Kompress" / "custom_data"
        elif WorkspaceExtension(workspace_extension) in {WorkspaceExtension.ADAPT}:
            return DockerPath.WORKSPACE.value / "Adapt" / "custom_data"

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
