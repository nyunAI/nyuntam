from datasets import load_dataset, DatasetDict
import os

ds = load_dataset("openai/gsm8k", "main")

assert isinstance(ds, DatasetDict), f"Expected datasets.DatasetDict, got {type(ds)}"
GSM_TEMPLATE = "Question: {question}\nAnswer:"


def restructure_fn(sample):
    sample["text"] = GSM_TEMPLATE.format(question=sample["question"])
    if "answer" in sample:
        sample["text"] += " " + sample["answer"]
    return sample


ds = ds.map(
    function=restructure_fn,
    batched=False,
    remove_columns=[],
    num_proc=os.cpu_count(),
    desc="Restructuring openai/gsm8k Dataset",
)
assert isinstance(ds, DatasetDict), f"Expected datasets.DatasetDict, got {type(ds)}"

save_path = "user_data/datasets/gsm8k_restructured"
if os.path.exists(save_path):
    print(f"[WARNING] Path {save_path} already exists. Overwriting...")
    os.system(f"rm -rf {save_path}")

os.makedirs(save_path, exist_ok=True)
ds.save_to_disk(save_path)
