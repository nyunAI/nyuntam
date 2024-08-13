from nyuntam.settings import ROOT
from nyuntam.utils.config import load_config, dump_config
from nyuntam.constants.keys import FactoryArgumentKeys, JobServices

# nyuntam_adapt
from nyuntam_adapt.utils import AdaptParams, create_instance

from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
import os
from argparse import Namespace, ArgumentParser


@dataclass
class Command(ABC):
    name: str = field(default="", init=False)
    arguments: List[Union[str, "Command"]] = field(default_factory=list)
    flags: Dict = field(default_factory=dict)

    _command: Optional[str] = None

    def build_command(self):
        command = f"{self.name}"
        for key, value in self.flags.items():
            if value is not None:
                command += f" --{key}={value}"
        for arg in self.arguments:
            if isinstance(arg, Command):
                command += f" {arg.get_command_as_subcommand()}"
            else:
                command += f" {arg}"
        return command

    @property
    def command(self):
        if self._command is None:
            self._command = self.build_command()
        return self._command

    def reset_command(self):
        self._command = None

    def reset_name(self):
        self.name = ""
        self.reset_command()

    def run(self):
        if os.system(self.command) != 0:
            raise ValueError(f"Failed to run command: {self.command}")

    @abstractmethod
    def get_command_as_subcommand(self):
        pass


@dataclass
class NyunRun(Command):
    name = f"python {str(ROOT / 'main.py')}"

    @classmethod
    def from_namespace(cls, args: Namespace):
        return cls(flags=vars(args))

    def get_command_as_subcommand(self):
        return self.command.replace("python", "").strip()


@dataclass
class NyunRunTorch(Command):
    name = "torchrun"
    flags = {"nnodes": 1, "nproc-per-node": 1}

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        num_gpu: Optional[int] = None,
        num_nodes: Optional[int] = None,
    ):
        nyun_run = NyunRun.from_namespace(args)

        num_nodes = num_nodes if num_nodes != None else cls.flags["nnodes"]
        num_gpu = num_gpu if num_gpu != None else cls.flags["nproc-per-node"]
        cls.flags.update({"nnodes": num_nodes, "nproc-per-node": num_gpu})
        return cls(arguments=[nyun_run], flags=cls.flags)

    def get_command_as_subcommand(self):
        return self.command


@dataclass
class NyunRunAccelerate(Command):
    name = "accelerate launch"
    flags = {"config_file": None}

    @classmethod
    def from_namespace(cls, args: Namespace):
        nyun_run = NyunRun.from_namespace(args)
        config = load_config(args.yaml_path or args.json_path)
        adapt_params = create_instance(AdaptParams, config)
        accelerate_config_path = dump_config(
            asdict(adapt_params.fsdp_args), ROOT / "accelerate_config.yaml"
        )
        accelerate_config_path = str(accelerate_config_path)
        del adapt_params
        del config
        return cls(
            arguments=[nyun_run],
            flags={"config_file": accelerate_config_path},
        )

    def get_command_as_subcommand(self):
        return self.command


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--yaml_path", type=str, help="Path to config (.yaml file)", default=None
    )
    parser.add_argument(
        "--json_path", type=str, help="Path to config (.json file)", default=None
    )
    args = parser.parse_args()

    assert (
        args.yaml_path is not None or args.json_path is not None
    ), f"No config file provided. Please specify either a YAML or JSON file."

    return args


def run():
    args = get_args()
    runner = NyunRun.from_namespace(args)
    runner.run()


def run_dist():
    args = get_args()

    config = load_config(args.yaml_path or args.json_path)

    job_service = JobServices(
        config.get(FactoryArgumentKeys.JOB_SERVICE, JobServices.KOMPRESS)
    )
    if job_service == JobServices.ADAPT:
        adapt_params = create_instance(AdaptParams, config)
        if adapt_params.DDP:
            num_gpu = len(adapt_params.cuda_id.split(","))
            runner = NyunRunTorch.from_args(args, num_gpu, adapt_params.num_nodes)
        elif adapt_params.FSDP:
            runner = NyunRunAccelerate.from_namespace(args)
        else:
            runner = NyunRun.from_namespace(args)

        del adapt_params
    elif job_service == JobServices.KOMPRESS:
        if config.get(FactoryArgumentKeys.ALGORITHM == "AQLM"):
            # Currently nnodes=1 for KOMPRESS
            num_gpu = len(config.get("CUDA_ID").split(","))
            runner = NyunRunTorch.from_args(args, num_gpu, 1)
        else:
            runner = NyunRun.from_namespace(args)

    del config
    del job_service

    runner.run()
