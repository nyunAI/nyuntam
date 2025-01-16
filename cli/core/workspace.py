import configparser

from pathlib import Path
from typing import Dict, AnyStr, Union, Tuple, Optional

from cli.core.constants import WorkspaceExtension, WorkspaceMessage, WorkspaceSpec
from cli.core.extension import (
    BaseExtension,
    KompressVisionExtension,
    KompressTextGenerationExtension,
    AdaptExtension,
)
from cli.core.logger import init_logger

from logging import getLogger

logger = getLogger(__name__)


class Workspace:
    def __init__(
        self,
        workspace_path: Path,
        custom_data_path: Path,
        extensions: WorkspaceExtension,
        overwrite: bool = False,
    ):

        self.workspace_path = workspace_path
        self.custom_data_path = custom_data_path
        self.extensions = extensions

        if not self.workspace_path.exists():
            logger.error(
                WorkspaceMessage.WORKSPACE_PATH_DOES_NOT_EXIST.format(
                    self.workspace_path
                )
            )
            raise ValueError(
                WorkspaceMessage.WORKSPACE_PATH_DOES_NOT_EXIST.format(
                    self.workspace_path
                )
            )

        if not WorkspaceSpec.get_workspace_spec_path(self.workspace_path).exists():
            Workspace.create_workspace_spec(
                self.workspace_path,
                self.custom_data_path,
                WorkspaceExtension.get_extensions_dict(self.extensions),
            )
            self.workspace_spec = Workspace.load_workspace_spec(self.workspace_path)
            logger.info(WorkspaceMessage.WORKSPACE_INITIALIZED)
        else:
            try:
                self.workspace_spec = Workspace.load_workspace_spec(self.workspace_path)

                if (
                    # check if the custom data path is different
                    self.workspace_spec.get(
                        WorkspaceSpec.CUSTOM_DATA, WorkspaceSpec.PATH
                    )
                    != str(self.custom_data_path.absolute().resolve())
                ):
                    if overwrite:
                        self.update_workspace_spec()
                        logger.info(
                            WorkspaceMessage.CUSTOM_DATA_PATH_UPDATED.format(
                                from_path=self.workspace_spec.get(
                                    WorkspaceSpec.CUSTOM_DATA, WorkspaceSpec.PATH
                                ),
                                to_path=self.custom_data_path.absolute().resolve(),
                            )
                        )
                        self.workspace_spec = Workspace.load_workspace_spec(
                            self.workspace_path
                        )
                    else:
                        raise ValueError(
                            WorkspaceMessage.CUSTOM_DATA_PATH_ALREADY_EXISTS.format(
                                current_path=self.workspace_spec.get(
                                    WorkspaceSpec.CUSTOM_DATA, WorkspaceSpec.PATH
                                ),
                                given_path=self.custom_data_path.absolute().resolve(),
                            )
                            + " Set overwrite=True to update the workspace spec.",
                        )

                if (
                    # check if the extensions are different
                    WorkspaceExtension.is_extension_different(
                        dict(self.workspace_spec[WorkspaceSpec.EXTENSIONS]),
                        WorkspaceExtension.get_extensions_dict(self.extensions),
                    )
                ):
                    if overwrite:
                        self.update_workspace_spec()
                        logger.info(
                            WorkspaceMessage.EXTENSION_UPDATED.format(
                                from_extensions=", ".join(
                                    WorkspaceExtension.get_extensions_list(
                                        dict(
                                            self.workspace_spec[
                                                WorkspaceSpec.EXTENSIONS
                                            ]
                                        )
                                    )
                                ),
                                to_extensions=", ".join(
                                    WorkspaceExtension.get_extensions_list(
                                        self.extensions
                                    )
                                ),
                            )
                        )
                        self.workspace_spec = Workspace.load_workspace_spec(
                            self.workspace_path
                        )
                    else:
                        raise ValueError(
                            WorkspaceMessage.EXTENSION_ALREADY_EXISTS.format(
                                current_extensions=", ".join(
                                    WorkspaceExtension.get_extensions_list(
                                        dict(
                                            self.workspace_spec[
                                                WorkspaceSpec.EXTENSIONS
                                            ]
                                        )
                                    )
                                ),
                                given_extensions=", ".join(
                                    WorkspaceExtension.get_extensions_list(
                                        self.extensions
                                    )
                                ),
                            )
                            + " Set overwrite=True to update the workspace spec."
                        )
                self.workspace_spec = Workspace.load_workspace_spec(self.workspace_path)
                logger.info(WorkspaceMessage.WORKSPACE_INITIALIZED)

            except FileNotFoundError:
                logger.info(
                    f"{WorkspaceMessage.WORKSPACE_SPEC_FOUND}"
                    " Creating a new workspace spec."
                )
                Workspace.create_workspace_spec(
                    self.workspace_path,
                    self.custom_data_path,
                    WorkspaceExtension.get_extensions_dict(self.extensions),
                )
                self.workspace_spec = Workspace.load_workspace_spec(self.workspace_path)

            except Exception as e:
                logger.error(e)
                raise e

    @staticmethod
    def init_logger(workspace_path: Path):
        init_logger(workspace_path)

    @staticmethod
    def create_workspace_spec(
        workspace_path: Path,
        custom_data_path: Path,
        extensions: Dict[WorkspaceExtension, bool],
    ):
        config = configparser.ConfigParser()
        config.update(
            {
                WorkspaceSpec.WORKSPACE: {
                    WorkspaceSpec.PATH: str(workspace_path.absolute().resolve())
                },
                WorkspaceSpec.CUSTOM_DATA: {
                    WorkspaceSpec.PATH: str(custom_data_path.absolute().resolve())
                },
                WorkspaceSpec.LOGS: {
                    WorkspaceSpec.PATH: str(
                        WorkspaceSpec.get_log_file_path(workspace_path)
                    )
                },
                WorkspaceSpec.EXTENSIONS: extensions,
            }
        )

        workspace_spec_dir = WorkspaceSpec.get_workspace_spec_dir(workspace_path)
        workspace_spec_dir.mkdir(parents=True, exist_ok=True)

        workspace_spec_path = WorkspaceSpec.get_workspace_spec_path(workspace_path)
        with open(workspace_spec_path, "w") as configfile:
            config.write(configfile)
        workspace_spec_path.chmod(0o444)

    def update_workspace_spec(self):
        # unlock the workspace spec file and update it
        workspace_spec_path = WorkspaceSpec.get_workspace_spec_path(self.workspace_path)
        workspace_spec_path.chmod(0o644)
        Workspace.create_workspace_spec(
            self.workspace_path,
            self.custom_data_path,
            WorkspaceExtension.get_extensions_dict(self.extensions),
        )

    @staticmethod
    def load_workspace_spec(workspace_path: Path):
        workspace_spec_path = WorkspaceSpec.get_workspace_spec_path(workspace_path)
        config = configparser.ConfigParser()
        config.read(workspace_spec_path)

        workspace_log_file_path = Path(
            config.get(WorkspaceSpec.LOGS, WorkspaceSpec.PATH)
        )
        # initialize the logger
        Workspace.init_logger(workspace_log_file_path)
        return config

    def __str__(self):
        return (
            f"Workspace: {self.workspace_path}\n"
            f"Custom Data: {self.custom_data_path}\n"
            f"Extension: {dict(self.workspace_spec[WorkspaceSpec.EXTENSIONS])}"
        )

    def __repr__(self):
        return self.__str__()

    def init_extension(self) -> BaseExtension:
        extensions = dict(self.workspace_spec[WorkspaceSpec.EXTENSIONS])
        ext_obj = BaseExtension()
        for key, value in extensions.items():
            if value == "True":
                if WorkspaceExtension(key) == WorkspaceExtension.VISION:
                    KompressVisionExtension()
                elif WorkspaceExtension(key) == WorkspaceExtension.TEXT_GENERATION:
                    KompressTextGenerationExtension()
                elif WorkspaceExtension(key) == WorkspaceExtension.ADAPT:
                    AdaptExtension()
        ext_obj.install()
        return ext_obj

    def get_workspace_env_file(self) -> Optional[Path]:
        return WorkspaceSpec.get_env_file_path(self.workspace_path)

    def init_workspace(self):
        """Initialize workspace directory structure"""
        # Create main directories
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        subdirs = ['models', 'datasets', 'jobs', 'logs']
        for subdir in subdirs:
            (self.workspace_path / subdir).mkdir(exist_ok=True)
        
        # Create custom_data directory if specified
        if self.custom_data_path:
            self.custom_data_path.mkdir(parents=True, exist_ok=True)


def get_workspace_and_custom_data_paths(
    workspace: Union[Path, AnyStr, None], custom_data: Union[Path, AnyStr, None]
) -> Tuple[Path, Path]:
    """
    Get the workspace and custom data paths from the provided input or workspace spec.

    Args:
        workspace (Union[Path, AnyStr, None]): The workspace path. If None, the current working directory is used.
        custom_data (Union[Path, AnyStr, None]): The custom data path. If None, it is derived from the workspace spec or set to "custom_data" under the workspace path.

    Returns:
        Tuple[Path, Path, WorkspaceExtension]: A tuple containing the workspace path and custom data path.
    """
    config = None
    extensions = None

    if workspace is None:
        workspace = Path.cwd()

    if custom_data is None:
        if WorkspaceSpec.get_workspace_spec_path(workspace).exists():
            config = Workspace.load_workspace_spec(workspace)
            custom_data = config.get(WorkspaceSpec.CUSTOM_DATA, WorkspaceSpec.PATH)
        else:
            custom_data = workspace / "custom_data"

    workspace_path = Path(workspace)
    custom_data_path = Path(custom_data)
    custom_data_path.mkdir(parents=True, exist_ok=True)

    if config:
        extensions = WorkspaceExtension.get_extensions_list(
            dict(config[WorkspaceSpec.EXTENSIONS])
        )

    return (workspace_path, custom_data_path, extensions)
