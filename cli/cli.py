import typer
from pathlib import Path
from version import __version__
from cli.docs import NYUN_TRADEMARK
from cli.core.workspace import (
    Workspace,
    WorkspaceExtension,  # noqa: F401
    get_workspace_and_custom_data_paths,
)

from docker.models.containers import Container
from docker.errors import ContainerError
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import List, Optional

SUPPORTED_SUFFIX = {".yaml", ".yml", ".json"}

app = typer.Typer()


@app.command()
def init(
    workspace_path: Optional[Path] = typer.Argument(
        None, help="Path to workspace directory. Defaults to current directory."
    ),
    custom_data_path: Optional[Path] = typer.Argument(
        None, help="Path to custom data directory. Defaults to workspace_path/custom_data"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", "-o", help="Overwrite existing workspace"
    ),
    extensions: List[str] = typer.Option(
        ["all"], "--extensions", "-e", help="Extensions to install"
    ),
):
    """
    Initialize the Nyun workspace and custom data directory.

    This command initializes the Nyun workspace and custom data directory.
    You can provide the path to the workspace directory and the custom data directory.
    If not provided, default paths will be used.
    Additionally, you can specify whether to overwrite the existing workspace spec and which extensions to install.
    """
    # Use current directory if no workspace_path provided
    if workspace_path is None:
        workspace_path = Path.cwd()
    
    # Use workspace_path/custom_data if no custom_data_path provided
    if custom_data_path is None:
        custom_data_path = workspace_path / "custom_data"
    
    try:
        workspace = Workspace(
            workspace_path=workspace_path,
            custom_data_path=custom_data_path,
            overwrite=overwrite,
            extensions=extensions[0],
        )
        workspace.init_extension()

        
        # Create standard directory structure
        # for dir_name in ["models", "datasets", "jobs", "logs", ".cache"]:
        #     (workspace_path / dir_name).mkdir(parents=True, exist_ok=True)
            
        # typer.echo(f"Workspace initialized at {workspace_path}")
        # typer.echo(f"Custom data directory at {custom_data_path}")
        
    except Exception as e:
        typer.echo(f"Failed to initialize workspace: {str(e)}")
        raise typer.Abort()


@app.command(help="Run scripts within the initialized Nyun workspace.")
def run(
    file_paths: List[Path] = typer.Argument(
        None, help="Path(s) to the YAML or JSON script file you want to run."
    ),
    log_path: Optional[Path] = typer.Option(None, help="Path to save Docker container logs"),
):
    """
    Run scripts within the initialized Nyun workspace.

    This command allows you to run scripts within the initialized Nyun workspace.
    You need to provide the path to the YAML or JSON script file you want to run.
    The script will be executed within the initialized workspace.
    """
    if not file_paths:
        typer.echo("Please provide the path(s) to the script file.")
        raise typer.Abort()

    if any(file_path.suffix not in SUPPORTED_SUFFIX for file_path in file_paths):
        typer.echo("All configs must be a .yaml or .json files")
        raise typer.Abort()

    # Get workspace paths and extensions
    workspace_path, custom_data_path, extensions = get_workspace_and_custom_data_paths(
        None, None
    )
    
    try:
        # Initialize workspace
        workspace = Workspace(
            workspace_path=workspace_path,
            custom_data_path=custom_data_path,
            overwrite=False,
            extensions=extensions[0],
        )
        # Initialize extensions
        ext_obj = workspace.init_extension()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            for file_path in file_paths:
                task = progress.add_task(
                    f"[white](Nyun) Running script {file_path}...",
                    total=1,
                    start=False,
                )
                
                # Stop progress to show container logs
                progress.stop()
                
                try:
                    running_container = ext_obj.run(
                        file_path=file_path, 
                        workspace=workspace,
                        log_path=log_path
                    )
                    
                    progress.start()
                    progress.update(
                        task,
                        advance=1,
                        description=f"[green]Successfully completed.",
                        completed=True,
                        refresh=True,
                    )
                except Exception as e:
                    progress.start()
                    progress.update(
                        task,
                        description=f"[red]Failed: {str(e)}",
                        completed=True,
                        refresh=True,
                    )
                    raise e
                
    except Exception as e:
        typer.echo(f"Failed: {str(e)}")
        raise typer.Abort()


@app.command(help="Show the version of the Nyun CLI.")
def version():
    """
    Show the version of the Nyun CLI.

    This command displays the version of the Nyun CLI currently installed on your system.
    """
    version_string = NYUN_TRADEMARK.format(version=__version__)
    typer.echo(version_string)
