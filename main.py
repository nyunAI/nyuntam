from settings import set_system_path

set_system_path()

from factory import Factory
from argparse import ArgumentParser
from utils.logger import set_logger

set_logger()

import logging


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


def main():
    args = get_args()
    config_path = args.yaml_path or args.json_path

    if config_path is None:
        raise ValueError(
            "No configuration file provided. Please specify either a YAML or JSON file."
        )

    try:
        if args.yaml_path:
            factory = Factory.create_from_yaml(args.yaml_path)
        else:
            factory = Factory.create_from_json(args.json_path)

    except Exception as e:
        logging.exception(f"Failed to create Factory instance: {e}")
        raise

    assert factory is not None, "Factory instance could not be created."

    logging.info(f"Running job with configuration: {factory.__class__.__name__}")

    try:
        factory.run()
        logging.info("Job completed successfully.")
    except Exception as e:
        logging.exception(f"Failed to run job: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Critical failure in main execution: {e}")
