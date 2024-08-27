from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM
from typing import Union
from argparse import ArgumentParser
from pathlib import Path
import torch
import gc


def free(object):
    del object
    gc.collect()
    torch.cuda.empty_cache()


def get_args():
    parser = ArgumentParser()
    # add arguments for basemodel and quantized model path
    parser.add_argument(
        "--base_model", type=str, help="Path to base model", default=None
    )
    parser.add_argument(
        "--quantized_model", type=str, help="Path to quantized model", default=None
    )
    parser.add_argument(
        "--tasks", type=str, help="Tasks to evaluate", default="gptq_wikitext:0"
    )
    parser.add_argument(
        "--results", type=str, help="Path to save results", default="./results"
    )
    # cache
    parser.add_argument(
        "--cache_dir", type=str, help="Path to cache directory", default=None
    )
    args = parser.parse_args()
    return args


def _evaluate_gptq_wikitext(
    model_name: str,
    model: Union[AutoModelForCausalLM, AutoAWQForCausalLM],
    tokenizer: AutoTokenizer,
    results_path: Path,
):

    from auto_gptq.utils import Perplexity
    import numpy as np

    # evaluate the model
    task = "gptq_wikitext"
    print(
        "========================================================================================================================\n"
        f"Running eval: model=({model_name}) task=({task})"
        "\n========================================================================================================================"
    )

    # calculate perplexity
    pp = Perplexity(model, tokenizer)
    pp_score = np.mean(pp.calculate_perplexity())
    print(f"Perplexity: {pp_score}")

    # save
    results_path = results_path / model_name / task
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / "results.txt", "w") as f:
        f.write(f"Perplexity: {pp_score}")


def _evaluate_lm_eval(
    model_name: str,
    model: Union[AutoModelForCausalLM, AutoAWQForCausalLM],
    tokenizer: AutoTokenizer,
    results_path: Path,
    task_and_num_fewshot: tuple,
):
    from lm_eval import simple_evaluate
    from lm_eval.utils import make_table
    from lm_eval.models.huggingface import HFLM

    # evaluate the model
    task, num_fewshot = task_and_num_fewshot
    eval_lm = HFLM(
        model_name=model_name, pretrained=model, tokenizer=tokenizer, batch_size=16
    )
    print(
        "========================================================================================================================\n"
        f"Running eval: model=({model_name}) task=({task}) nshot=({num_fewshot})"
        "\n========================================================================================================================"
    )
    eval_results = simple_evaluate(
        eval_lm, tasks=[task], num_fewshot=num_fewshot, batch_size=16
    )
    free(eval_lm)
    tabular = make_table(eval_results)
    print(tabular)
    # save
    results_path = results_path / model_name / task
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"results_{num_fewshot}.txt", "w") as f:
        f.write(tabular)


def evaluate(
    model_name: str,
    model: Union[AutoModelForCausalLM, AutoAWQForCausalLM],
    tokenizer: AutoTokenizer,
    results_path: Path,
    tasks: list,
):
    # evaluate the model
    for task, num_fewshot in tasks:
        if "gptq_wikitext" in task:
            _evaluate_gptq_wikitext(model_name, model, tokenizer, results_path)
        else:
            print(
                "WARNING: We suggest to use lm_eval's cli for evaluating tasks other than gptq_wikitext."
            )
            _evaluate_lm_eval(
                model_name, model, tokenizer, results_path, (task, num_fewshot)
            )


if __name__ == "__main__":
    args = get_args()
    tasks = args.tasks.split(",")
    tasks = [(task.split(":")[0], int(task.split(":")[1])) for task in tasks]

    # evaluate base model
    if args.base_model:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, cache_dir=args.cache_dir, device_map="auto"
        )
        base_tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, cache_dir=args.cache_dir
        )
        evaluate("base", base_model, base_tokenizer, Path(args.results), tasks)
        free(base_model)
        free(base_tokenizer)

    if args.quantized_model:
        # evaluate quantized model
        quantized_model = AutoAWQForCausalLM.from_quantized(
            args.quantized_model, cache_dir=args.cache_dir
        )
        quantized_tokenizer = AutoTokenizer.from_pretrained(
            args.quantized_model, cache_dir=args.cache_dir
        )
        if not hasattr(quantized_model, "device"):
            setattr(quantized_model, "device", quantized_model.model.device)
        evaluate(
            "quantized", quantized_model, quantized_tokenizer, Path(args.results), tasks
        )
        free(quantized_model)
        free(quantized_tokenizer)
