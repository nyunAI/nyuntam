import sys
from pathlib import Path
import dataclasses
import typing as tp
import json

this_dir = Path(__file__).resolve().parent
sys.path.append(str(this_dir.parent))

from .main import (
    Engine,
    STTInput,
    EngineEnvironmentConfig,
    parse_args,
    print_dict,
    DEFAULT_CONFIG,
)

AUDIO_SAMPLES = "/path/to/audio_samples"


@dataclasses.dataclass
class BenchmarkSampleResult:
    latency: float
    ttfs: float
    audio_file_path: str
    stt_latency: float = 0.0

    @property
    def audio_length(self) -> int:
        return int(self.audio_file_path.split("/")[-2][:-3])


if __name__ == "__main__":
    args = parse_args()
    if args.config is None:
        args.config = DEFAULT_CONFIG
    config = EngineEnvironmentConfig.from_yaml(args.config)
    print(f"Environment config:")
    print_dict(config.to_dict())

    engine = Engine(config)
    data: tp.List[tp.Dict[str, tp.Any]] = []
    try:
        for audio_file in Path(AUDIO_SAMPLES).rglob("**/*.wav"):
            stt_input = STTInput(environment_config=config.stt, audio_path=audio_file)
            response = engine.call(stt_input)
            sample = BenchmarkSampleResult(
                latency=response.latency,
                ttfs=response.llm_response.ttfs,
                audio_file_path=str(audio_file),
                stt_latency=response.stt_latency,
            )
            print_dict(
                {
                    **dataclasses.asdict(sample),
                    "audio_length": sample.audio_length,
                }
            )
            print(f"-" * 50)
            data.append(
                {
                    **dataclasses.asdict(sample),
                    "audio_length": sample.audio_length,
                }
            )

        with open("benchmark.json", "w") as f:
            json.dump(data, f)
    except Exception as e:
        engine.terminate()
        raise e
