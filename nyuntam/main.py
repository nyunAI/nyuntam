from settings import set_system_path

set_system_path()

from nyuntam.factory import Factory
from nyuntam.utils.logger import set_logger
from nyuntam.commands import get_args

set_logger()

import logging


def main():
    args = get_args()
    try:
        if args.yaml_path:
            factory = Factory.create_from_yaml(args.yaml_path)
        else:
            factory = Factory.create_from_json(args.json_path)

    except Exception as e:
        logging.exception(f"Failed to create Factory instance: {e}")
        raise

    assert factory is not None, "Factory instance could not be created."

    logging.info(
        f"Running job with configuration: {factory.algorithm.__class__.__name__}"
    )

    try:
        factory.run()
        logging.info("Job completed successfully.")
    except Exception as e:
        logging.exception(f"Failed to run job: {e}")
        raise


if __name__ == "__main__":
    main()
