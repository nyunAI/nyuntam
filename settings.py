from pathlib import Path
from typing import Set, Optional, Union, List, Dict
import sys

ROOT = Path(__file__).parent

# SUBMODULES = {"submodule-name": {"submodule-algorithm": ["paths/to/add"]}}
SUBMODULES: Dict[str, Dict[str, List[Union[str, Path]]]] = {
    "text_generation": {"pruning": ["flap"]},
    "nyuntam_adapt": {
        "tasks": [
            "object_detection_mmdet/mmdetection",
            "image_segmentation_mmseg/mmsegmentation",
            "pose_estimation_mmpose/mmpose",
        ]
    },
    # "vision": {},
}


def set_system_path(*paths: Optional[Union[str, Path]]):
    """Set the system path to include the path to the modules.

    Args:
        paths: Path to the module.
    """

    paths_to_add: Set[Union[str, Path]] = set()

    # root
    paths_to_add.add(ROOT)
    paths_to_add.add(ROOT.parent)

    for submodule, algorithm in SUBMODULES.items():
        # submodules
        paths_to_add.add(ROOT / submodule)
        for algorithm_type, algorithm_paths in algorithm.items():
            # submodule-algorithm-paths
            for path in algorithm_paths:
                paths_to_add.add(ROOT / submodule / algorithm_type / path)

    # additional paths
    if paths:
        print("PATHHHHHH", paths)
        paths_to_add.update(paths)

    for path in paths_to_add:
        sys.path.append(str(path))
