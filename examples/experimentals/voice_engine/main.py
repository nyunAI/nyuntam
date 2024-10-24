from dataclasses import dataclass, field, asdict, is_dataclass, fields
from argparse import ArgumentParser
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
import typing as tp
import subprocess
import queue
import threading
from abc import ABC, abstractmethod
import yaml
import json
import requests
import logging
import time

LOGGER = None
DEFAULT_CONFIG = "recipe/default.yaml"


def set_logger(*args, **kwargs):
    global LOGGER
    logging.basicConfig(*args, **kwargs)
    LOGGER = logging.getLogger(__name__)


##################################################
#           Environment Configurations           #
##################################################


class EnvironmentConfigMeta(ABC):

    @classmethod
    def from_dict(cls, data: dict):
        kwargs = {}
        for field in fields(cls):
            if field.name in data:
                if is_dataclass(field.type):
                    kwargs[field.name] = field.type.from_dict(data[field.name])
                else:
                    kwargs[field.name] = data[field.name]
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)

    def to_dict(self):
        return asdict(self)

    def to_yaml(self, path: str):
        with open(path, "w") as file:
            yaml.dump(self.to_dict(), file)


@dataclass
class EnvironmentConfig(ABC):
    n_threads: int = field(default=4)
    n_procs: int = field(default=1)
    flash_attn: bool = field(default=True)
    gpu: bool = field(default=False)
    port: int = field(default=8080)
    model: tp.Union[str, Path] = field(default="")

    _executable: tp.Union[str, Path] = field(default="")
    _warmup: int = field(default=0)

    @property
    def executable_path(self) -> str:
        if isinstance(self._executable, Path):
            return str(self._executable.absolute().resolve())
        return self._executable

    @property
    def model_path(self) -> str:
        if isinstance(self.model, Path):
            return str(self.model.absolute().resolve())
        return self.model

    def get_model_option(self) -> str:
        return f"-m {self.model_path}"

    @abstractmethod
    def get_options(self) -> str:
        pass

    def __post_init__(self):
        assert self.n_threads in [4, 2, 1], "Number of threads must be 1, 2, or 4"
        assert self.n_procs in [1, 2, 4], "Number of processes must be 1, 2, or 4"
        assert self.port > 0, "Port number must be positive"


@dataclass
class STTEnvironmentConfig(EnvironmentConfig, EnvironmentConfigMeta):
    """Environment configuration for whisper.cpp"""

    port: int = field(default=8080)
    _warmup: int = field(default=5)

    def get_options(self):
        SPACE = " "
        flash_attn = "-fa"
        gpu = "-ng"
        threads = f"-t {self.n_threads}"
        processes = f"-p {self.n_procs}"
        port = f"--port {self.port}"
        cmd = threads
        cmd += SPACE
        cmd += processes
        cmd += SPACE
        if not self.gpu:
            cmd += gpu
            cmd += SPACE
        if self.flash_attn:
            cmd += flash_attn
            cmd += SPACE
        cmd += port
        cmd += SPACE
        return cmd


@dataclass
class LLMEnvironmentConfig(EnvironmentConfig, EnvironmentConfigMeta):
    """Environment configuration for llama.cpp"""

    batch_size: int = field(default=8192)
    ubatch_size: int = field(default=512)
    n_predict: int = field(default=-1)
    stream: bool = field(default=True)
    port: int = field(default=8081)
    _warmup: int = field(default=15)

    def get_options(self):
        SPACE = " "
        flash_attn = "-fa"
        threads = f"-t {self.n_threads}"
        batch_size = f"-b {self.batch_size}"
        ubatch_size = f"-ub {self.ubatch_size}"
        port = f"--port {self.port}"
        n_predict = f"-n {self.n_predict}"

        cmd = threads
        cmd += SPACE
        cmd += batch_size
        cmd += SPACE
        cmd += ubatch_size
        cmd += SPACE
        cmd += n_predict
        cmd += SPACE
        if self.flash_attn:
            cmd += flash_attn
            cmd += SPACE
        cmd += port
        cmd += SPACE
        return cmd


@dataclass
class EngineEnvironmentConfig(EnvironmentConfigMeta):
    stt: STTEnvironmentConfig = field(default_factory=STTEnvironmentConfig)
    llm: LLMEnvironmentConfig = field(default_factory=LLMEnvironmentConfig)
    log_path: tp.Union[str, Path] = field(default="environment.log")

    def __post_init__(self):
        set_logger(
            filename=self.log_path,
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


##################################################
#           Argument Parsing Functions           #
##################################################


def parse_args():
    # TODO: Add more arguments as necessary
    parser = ArgumentParser(description="Environment Configuration Parser")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file (.yaml)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="environment.log",
        help="Path to the log file",
    )
    return parser.parse_args()


##################################################
#                  Engine Class                  #
##################################################


@dataclass
class STTInput:
    environment_config: STTEnvironmentConfig
    audio_path: tp.Union[str, Path]
    data: tp.Optional[tp.Dict[str, tp.Any]] = None

    @property
    def config(self) -> STTEnvironmentConfig:
        return self.environment_config

    @property
    def audio(self) -> str:
        if isinstance(self.audio_path, Path):
            return str(self.audio_path.absolute().resolve())
        return self.audio_path


@dataclass
class STTResponse:
    text: str

    @classmethod
    def from_response(cls, response: requests.Response):
        if response.status_code != 200:
            raise_exception_from_response(response)
        return cls(response.json()["text"])


def default_llm_input_data_factory():
    return {
        "prompt": "",
        "n_predict": -1,
        "stream": True,
    }


@dataclass
class LLMInput:
    environment_config: LLMEnvironmentConfig
    prompt: str
    data: tp.Optional[tp.Dict[str, tp.Any]] = field(
        default_factory=default_llm_input_data_factory
    )

    @property
    def stream(self) -> bool:
        if self.data is not None and "stream" in self.data:
            return self.data["stream"]
        return self.config.stream

    @stream.setter
    def stream(self, value: bool):
        if self.data is None:
            self.data = {}
        self.data["stream"] = value

    @property
    def config(self) -> LLMEnvironmentConfig:
        return self.environment_config

    @classmethod
    def from_stt_response(
        cls, environment_config: LLMEnvironmentConfig, stt_response: STTResponse
    ):
        return cls(environment_config, stt_response.text)

    def get_data(self):
        return {
            **self.data,
            "prompt": self.prompt,
        }


@dataclass
class LLMResponse:
    text: str
    streams: tp.List[tp.Dict[str, tp.Any]] = field(default_factory=list)
    ttfs: float = 0.0
    stream: bool = False


@dataclass
class EngineInput:
    stt_input: tp.Optional[STTInput] = None
    llm_input: tp.Optional[LLMInput] = None


@dataclass
class EngineResponse:
    stt_response: tp.Optional[STTResponse] = None
    llm_response: tp.Optional[LLMResponse] = None
    latency: float = 0.0
    stt_latency: float = 0.0


class Engine:
    def __init__(self, config: EngineEnvironmentConfig):
        self.config = config
        self.init_handlers()

    @property
    def stt(self) -> tp.Optional[subprocess.Popen]:
        if hasattr(self, "_stt_process"):
            return self._stt_process
        else:
            return None

    @stt.setter
    def stt(self, value: subprocess.Popen):
        self._stt_process = value

    @property
    def llm(self) -> tp.Optional[subprocess.Popen]:
        if hasattr(self, "_llm_process"):
            return self._llm_process
        else:
            return None

    @llm.setter
    def llm(self, value: subprocess.Popen):
        self._llm_process = value

    def init_handlers(self) -> None:
        # Initialize stt
        if self.stt is not None:
            raise ValueError("STT process already initialized")
        try:
            self.stt = initialize_stt_environment(self.config.stt)
            LOGGER.info(f"STT process initialized with PID: {self.stt.pid}")
        except Exception as e:
            LOGGER.error(f"Failed to initialize STT process: {e}")
            raise e

        # Initialize llm
        if self.llm is not None:
            raise ValueError("LLM process already initialized")
        try:
            self.llm = initialize_llm_environment(self.config.llm)
            if self.llm.poll() is not None:
                raise Exception(f"LLM process failed to start: {self.llm.stderr}")
            LOGGER.info(f"LLM process initialized with PID: {self.llm.pid}")
        except Exception as e:
            LOGGER.error(f"Failed to initialize LLM process: {e}")
            raise e

        # NOTE: When using chain of responsibility, initialize handlers here

    def call(self, input: STTInput) -> EngineResponse:
        assert isinstance(input, STTInput), "Input must be of type STTInput"
        tick = time.time()
        stt_response = STTResponse.from_response(call_stt_environment(input))
        stt_latency = time.time()
        LOGGER.debug(f"STT response: {stt_response}")
        llm_input = LLMInput.from_stt_response(self.config.llm, stt_response)
        if llm_input.stream:
            # implement stream response handling
            decoded_streams = queue.Queue()
            stream_queue = queue.Queue()
            stop_event = threading.Event()
            decode_thread = threading.Thread(
                target=decode_stream,
                args=(stop_event, stream_queue, decoded_streams, True),
            )
            decode_thread.start()
            llm_input.data["stream"] = True
            ttfs = None
            response = call_llm_environment(llm_input)
            LOGGER.debug(f"LLM response: {response}")
            if not response.ok or response.status_code != 200:
                raise_exception_from_response(response)
            for line in response.iter_lines():
                if line:
                    if ttfs is None:
                        ttfs = time.time()
                        print("TTFS: ", ttfs - tick)

                    stream_queue.put(line)
            tock = time.time()
            stop_event.set()
            decode_thread.join()
            llm_response = LLMResponse(
                text=decoded_streams_to_text(list(decoded_streams.queue)),
                streams=list(decoded_streams.queue),
                ttfs=ttfs - tick,
                stream=True,
            )
            return EngineResponse(
                stt_response=stt_response,
                llm_response=llm_response,
                latency=tock - tick,
                stt_latency=stt_latency - tick,
            )
        else:
            raise NotImplementedError("Non-streaming response handling not implemented")
        # NOTE: When using chain of responsibility, call handlers here

    def terminate(self):
        if self.stt is not None:
            kill_process(self.stt)
        if self.llm is not None:
            kill_process(self.llm)


##################################################
#                 Utility Functions              #
##################################################


@contextmanager
def warmup_environment(warmup_time: int):
    yield
    time.sleep(warmup_time)


def raise_exception_from_response(response: requests.Response):
    LOGGER.error(
        f"API call failed with status code: {response.status_code}, response: {response.text}"
    )
    raise Exception(
        f"API call failed with status code: {response.status_code}, response: {response.text}"
    )


def kill_process(process: subprocess.Popen):
    process.terminate()
    process.wait()
    LOGGER.info(f"Process {process.pid} terminated gracefully.")


def initialize_environment(config: EnvironmentConfig):
    with warmup_environment(config._warmup):
        cmd: tp.List[str] = (
            [config.executable_path]
            + config.get_options().split()
            + config.get_model_option().split()
        )
        LOGGER.info(f"Initializing environment with command: {' '.join(cmd)}")
        return subprocess.Popen(cmd)


def initialize_stt_environment(config: STTEnvironmentConfig):
    proc = initialize_environment(config=config)
    url = f"http://127.0.0.1:{config.port}/load"
    data = {"model": config.model_path}
    response = requests.post(url, json=data)
    if not response.ok or response.status_code != 200:
        LOGGER.error(f"Failed to load STT model.")
        raise_exception_from_response(response)
    return proc


initialize_llm_environment: tp.Callable[[LLMEnvironmentConfig], subprocess.Popen] = (
    initialize_environment
)


def call_stt_environment(input: STTInput):
    url = f"http://127.0.0.1:{input.config.port}/inference"
    files = {"file": open(input.audio, "rb")}
    data = input.data
    return requests.post(url, files=files, data=data)


def call_llm_environment(input: LLMInput):
    url = f"http://127.0.0.1:{input.config.port}/completion"
    data = input.get_data()
    return requests.post(url, json=data, stream=input.stream)


def decode_stream(
    stop_event: threading.Event,
    stream_queue: queue.Queue,
    decoded_streams: queue.Queue,
    decode_and_print: bool = False,
):
    while not stop_event.is_set() or not stream_queue.empty():
        try:
            line: bytearray = stream_queue.get(
                timeout=0.001
            )  # Get stream data from queue
            if line:
                json_response = json.loads(line.decode("utf-8").replace("data: ", ""))
                decoded_streams.put(json_response)
                # if decode_and_print:
                #     # print decoded stream continously with flush
                #     print(json_response["content"], end="", flush=True)
        except queue.Empty:
            pass  # No data to process yet, continue


def decoded_streams_to_text(decoded_streams: tp.List[tp.Dict[str, tp.Any]]) -> str:
    return " ".join([stream["content"] for stream in decoded_streams])



##################################################
##################################################

def print_dict(d: dict, indent: int = 0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(" " * indent, f" - {k}:")
            print_dict(v, indent + 4)
        else:
            print(" " * indent, f" - {k}: {v}")


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        config = EngineEnvironmentConfig.from_yaml(args.config)
        print_dict(config.to_dict())

    else:
        config = EngineEnvironmentConfig()
        config.to_yaml("/home/piuser/edge/recipe/default.yaml")

    engine = Engine(config)

    try:
        while True:
            # input an audio file path from the user
            user_input = input("Enter the path to the audio file: ")
            if user_input == "":
                user_input = "/home/piuser/shwu/audio_samples/5sec/79833.wav"
            if user_input == "exit":
                break
            stt_input = STTInput(environment_config=config.stt, audio_path=user_input)
            response = engine.call(stt_input)
            print_dict(
                {
                    "latency": response.latency,
                    "ttfs": response.llm_response.ttfs,
                    "stt_latency": response.stt_latency,
                }
            )
            print(f"-" * 50)
    except Exception as e:
        engine.terminate()
        raise e