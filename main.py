from settings import set_system_path

set_system_path()

from factory import Factory
from argparse import ArgumentParser
import logging

# main.py is to be run as python main.py --yaml_path path/to/yaml  (or) python main.py --json_path path/to/json


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--yaml_path", type=str, help="Path to config (.yaml file)", default=None
    )
    parser.add_argument(
        "--json_path", type=str, help="Path to config (.json file)", default=None
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print("Args: ", args)
    if args.yaml_path is None and args.json_path is None:
        raise ValueError("No config file provided.")

    if args.yaml_path is not None:
        factory = Factory.create_from_yaml(args.yaml_path)
    elif args.json_path is not None:
        factory = Factory.create_from_json(args.json_path)
    else:
        raise ValueError("No config file provided.")

    if factory is None:
        raise ValueError("Factory instance could not be created.")

    logger = logging.getLogger(__name__)
    logger.info(f"Running job: {factory}")
    try:
        factory.run()
    except Exception as e:
        logger.exception(f"Failed to run job: {e}")
