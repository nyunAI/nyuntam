import logging
import os
from pathlib import Path

from nyuntam.factory import Factory
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--yaml_path", type=str, help="Path to config (.yaml file)", default=None)
    parser.add_argument("--json_path", type=str, help="Path to config (.json file)", default=None)

    # Optional CLI overrides
    parser.add_argument("--logging_path", type=str, help="Override logging path", default=None)
    parser.add_argument("--job_id", type=str, help="Override job ID", default=None)
    parser.add_argument("--cache_path", type=str, help="Override cache path", default=None)
    parser.add_argument("--data_path", type=str, help="Override data path", default=None)
    parser.add_argument("--output_path", type=str, help="Override output path", default=None)

    args = parser.parse_args()

    # At least one config file must be provided
    if not args.yaml_path and not args.json_path:
        raise ValueError("No config file provided. Please specify either --yaml_path or --json_path.")

    return args

def main():
    args = get_args()
    try:
        # Build the factory
        if args.yaml_path:
            factory = Factory.create_from_yaml(args.yaml_path)
        else:
            factory = Factory.create_from_json(args.json_path)
    except Exception as exc:
        logging.exception(f"Failed to create Factory instance: {exc}")
        raise

    assert factory is not None, "Factory instance could not be created."

    # Override config keys if CLI arguments are provided
    if args.logging_path is not None:
        factory.config["LOGGING_PATH"] = args.logging_path

    if args.job_id is not None:
        factory.config["JOB_ID"] = args.job_id

    if args.cache_path is not None:
        factory.config["CACHE_PATH"] = args.cache_path

    if args.data_path is not None:
        factory.config["DATA_PATH"] = args.data_path

    if args.output_path is not None:
        factory.config["OUTPUT_PATH"] = args.output_path

    # Create the job_id folder as the top-level directory
    job_id = factory.config.get("JOB_ID", "default_job")
    base_folder = Path(job_id)
    base_folder.mkdir(exist_ok=True, parents=True)

    # Inside job_id, create a "user_data" subfolder and then any other subfolders
    user_data_folder = base_folder / "user_data"
    user_data_folder.mkdir(exist_ok=True)

    models_folder = user_data_folder / "models"
    datasets_folder = user_data_folder / "datasets"
    jobs_folder = user_data_folder / "jobs"
    logs_folder = user_data_folder / "logs"
    cache_folder = user_data_folder / ".cache"

    models_folder.mkdir(exist_ok=True)
    datasets_folder.mkdir(exist_ok=True)
    jobs_folder.mkdir(exist_ok=True)
    logs_folder.mkdir(exist_ok=True)
    cache_folder.mkdir(exist_ok=True)

    # Update config with new directories
    factory.config["USER_FOLDER"] = str(user_data_folder)
    factory.config["MODEL_PATH"] = str(models_folder)
    factory.config["DATASET_PATH"] = str(datasets_folder)
    factory.config["JOB_PATH"] = str(jobs_folder)
    factory.config["LOGGING_PATH"] = str(logs_folder)
    factory.config["CACHE_PATH"] = str(cache_folder)

    logging.info(f"Running job with configuration: {factory.algorithm.__class__.__name__}")

    try:
        factory.run()
        logging.info("Job completed successfully.")
    except Exception as exc:
        logging.exception(f"Failed to run job: {exc}")
        raise

if __name__ == "__main__":
    main()
