"""
This module provides utility functions for managing Docker containers and images.
It includes functions for pulling Docker images, starting containers, running containers,
and removing containers.
"""

from typing import Union, Dict, Optional
from logging import getLogger
import os
import docker
from dotenv import load_dotenv, dotenv_values
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn
from docker.models.containers import Container, ExecResult
from cli.core.constants import (
    DockerPath,
    DockerCommand,
    WorkspaceExtension,
    NYUN_ENV_KEY_PREFIX,
    EMPTY_STRING,
)
from cli import NYUNTAM
from cli import SERVICES as NyunServices
from docker.types import Mount, DeviceRequest
from docker.errors import NotFound, ImageNotFound, ContainerError
from pathlib import Path


logger = getLogger(__name__)

# ========================================
#               Docker Utils
# ========================================


def get_docker_client() -> docker.DockerClient:
    """
    Get a Docker client instance authenticated with the provided credentials.

    Returns:
        docker.DockerClient: A Docker client instance.
    """
    load_dotenv()

    client = docker.from_env()
    try:
        client.login(
            username=os.getenv("DOCKER_USERNAME"),
            password=os.getenv("DOCKER_ACCESS_TOKEN"),
        )
    except Exception as e:
        logger.error(
            "Failed to authenticate with Docker credentials. Only public images can be pulled."
        )
    return client


# TODO: add argument silent: bool = False to suppress loading outputs for run commands.
def pull_docker_image(*image: "NyunDocker"):
    """
    Pull Docker images in parallel using ThreadPoolExecutor.

    Args:
        *image (NyunDocker): One or more NyunDocker instances representing the Docker images to pull.

    Raises:
        Exception: If any of the images fail to pull.
    """
    client = get_docker_client()

    total = len(image)
    count = 0
    img_map = {}

    def counter(img):
        nonlocal img_map
        nonlocal count

        # return from memoized
        if img_map.get(str(img), False):
            return img_map.get(str(img))

        if count == total:
            count = 0
        count += 1
        img_map[str(img)] = count  # memoize
        return count

    def wrap(repo, tag, task):
        try:
            # check if image already exists (exclude dangling images)
            if client.images.list(repo, tag, filters={"dangling": True}):
                return
            client.images.pull(repo, tag)
        except ImageNotFound as e:
            raise ImageNotFound(
                f'Access denied. Reach out to us at "contact@nyunai.com" for access'
            )
        except Exception as e:
            raise Exception(f"Failed to pull") from e

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    )
    with progress:
        tasks = {
            progress.add_task(
                f"[white]Component [{counter(img)}/{total}] loading ({img.repository}:{img.tag}).",
                total=total,
                start=False,
            ): img
            for img in image
        }

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(wrap, img.repository, img.tag, tasks[task]): task
                for task, img in tasks.items()
            }

            for future in as_completed(futures):
                task = futures[future]
                img = tasks[task]

                try:
                    future.result()
                    progress.update(
                        task,
                        advance=1,
                        description=f"[green]Component [{counter(img)}/{total}] loaded ({img.repository}:{img.tag}).",
                        completed=True,
                        refresh=True,
                    )
                except Exception as e:
                    progress.update(
                        task,
                        advance=1,
                        description=f"[red]Component [{counter(img)}/{total}] failed to load ({img.repository}:{img.tag}). {e}.",
                        completed=True,
                        refresh=True,
                    )
                    logger.exception(
                        f"Failed to pull ({img.repository}:{img.tag}). {e}."
                    )


def remove_docker_image(*image: "NyunDocker"):
    """
    Remove Docker images.

    Args:
        *image (NyunDocker): One or more NyunDocker instances representing the Docker images to remove.

    Raises:
        Exception: If any of the images fail to remove.
    """
    client = get_docker_client()
    for img in image:
        try:
            client.images.remove(img.repository, img.tag)
            print(f"Image {img} removed successfully")
        except Exception as e:
            logger.error(f"Image {img} failed to remove: {e}")
            raise Exception from e


def start_docker_container(*image: "NyunDocker"):
    """
    Start Docker containers in parallel using ThreadPoolExecutor.

    Args:
        *image (NyunDocker): One or more NyunDocker instances representing the Docker images to start.

    Raises:
        Exception: If any of the containers fail to start.
    """
    client = get_docker_client()

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(client.containers.crea, img.repository, img.tag): img
            for img in image
        }

        for future in futures:
            img = futures[future]
            try:
                future.result()
                print(f"Container {img} started successfully")
            except Exception as e:
                logger.error(f"Container {img} failed to start: {e}")
                raise Exception from e


def run_docker_container(
    script: Path,
    workspace: "Workspace",
    metadata: "DockerMetadata",
    *image: "NyunDocker",
    log_path: Optional[Path] = None,
) -> Container:
    """
    Run a Docker container with a specified command in detached mode with auto-removal.

    Args:
        script (Path): The script path to run in the Docker container. (It will be mounted on the docker inside "/scripts").
        workspace (Workspace): The workspace object.
        metadata (DockerMetadata): The docker metadata object.
        *image (NyunDocker): A NyunDocker instance representing the Docker image to run.
        log_path (Optional[Path]): The path to save Docker container logs.

    Returns:
        Container: The running Docker container.

    Raises:
        Exception: If the container fails to run.
    """

    # TODO: Update to handle running multiple containers

    try:
        client = get_docker_client()
        script_path = DockerPath.get_script_path_in_docker(script_path=script)
        command = DockerCommand.get_run_command(script_path=script_path)
        service = get_service_from_metadata_extension_type(
            extension_type=metadata.extension_type
        )
        mounts = [
            # Mount workspace dir with write permissions
            Mount(
                source=str(workspace.workspace_path),
                target=str(DockerPath.USER_DATA.value),
                type="bind",
                read_only=False,
            ),
            # Mount custom data dir
            Mount(
                source=str(workspace.custom_data_path),
                target=str(DockerPath.CUSTOM_DATA.value),
                type="bind",
                read_only=True,
            ),
            # Mount script
            Mount(
                source=str(script.absolute().resolve()),
                target=str(script_path),
                type="bind",
                read_only=True,
            ),
            # Mount nyuntam directory
            Mount(
                source=str(NYUNTAM),
                target=str(DockerPath.NYUNTAM.value),
                type="bind",
                read_only=True,
            ),
        ]

        environment = get_environment_keys_from_workspace(workspace.get_workspace_env_file()) if workspace.get_workspace_env_file() else None
        device_requests = [DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])]
        
        # Change working directory to user_data which is writable
        working_dir = str(DockerPath.USER_DATA.value)

        print(f"Running {image[0]} with command: {command}")
        print(f"Mounts: {mounts}")
        print(f"Environment: {environment}")
        print(f"Device Requests: {device_requests}")
        print(f"Working Dir: {working_dir}")

        # Run container silently
        running_container: Container = client.containers.run(
            command=command,
            image=str(image[0]),
            device_requests=device_requests,
            detach=True,
            mounts=mounts,
            remove=False,
            working_dir=working_dir,
            environment=environment,
            stdin_open=True,
            tty=True,
        )

        # Wait for container to finish
        result = running_container.wait()
        if result['StatusCode'] != 0:
            raise Exception(f"Container execution failed")

        # Clean up container
        try:
            running_container.remove()
        except Exception as e:
            print(f"Warning: Could not remove container: {e}")

        # Write logs if path provided
        if log_path:
            try:
                # Create directory if it doesn't exist
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Open file in append mode to create it if it doesn't exist
                with open(log_path, 'a') as f:
                    f.write(f"\n--- Container logs for {metadata.algorithm} started ---\n")
                    for log in running_container.logs(stream=True, follow=True):
                        f.write(f"{log.decode().strip()}\n")
                        f.flush()
                    f.write(f"--- Container logs ended ---\n")
            except Exception as e:
                logger.error(f"Failed to write logs to {log_path}: {str(e)}")
                print(f"Warning: Could not write logs to {log_path}")

        return running_container

    except Exception as e:
        logger.error(f"Failed to run {metadata.algorithm}: {str(e)}")
        raise Exception from e


def remove_container(*image: "NyunDocker"):
    """
    Remove a Docker container.

    Args:
        *image (NyunDocker): A NyunDocker instance representing the Docker container to remove.

    Raises:
        Exception: If the container fails to remove.
    """
    client = get_docker_client()
    try:
        container_to_remove: Container = client.containers.get(
            container_id=image.container_id
        )
        container_to_remove.remove()
        print(f"Container {image} removed successfully")
    except Exception as e:
        logger.error(f"Container {image} failed to remove: {e}")
        raise Exception from e


def get_environment_keys_from_workspace(env_file_path: Path) -> Dict[str, str]:
    """
    Get the environment keys from the workspace.

    Args:
        workspace (Workspace): The workspace instance.

    Returns:
        Dict[str, str]: A dictionary containing the environment keys.
    """

    print(f"Reading environment keys from {env_file_path}")
    return {
        key.replace(NYUN_ENV_KEY_PREFIX, EMPTY_STRING): value
        for key, value in dotenv_values(env_file_path).items()
        if key.startswith(NYUN_ENV_KEY_PREFIX)
    }


def get_service_from_metadata_extension_type(extension_type: WorkspaceExtension) -> str:
    """Get the service name from the metadata extension type."""
    if extension_type in {
        WorkspaceExtension.TEXT_GENERATION,
        WorkspaceExtension.VISION,
        WorkspaceExtension.ADAPT,
    }:
        return str(NYUNTAM)

    raise ValueError(f"Invalid extension type: {extension_type}")
