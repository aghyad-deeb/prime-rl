import glob
import os

import pandas as pd

main_dir = os.path.expanduser("~/reward_seeker/environments/verl_envs")
dataset_name = "omit_description"
dataset_path = os.path.join(main_dir, dataset_name)
for path in glob.glob(f"{dataset_path}/data*.parquet"):
    df = pd.read_parquet(path)
df["task"] = df["data_source"]
df["answer"] = df["ground_truth"]
df["info"] = df["extra_info"]
df.drop(columns=["data_source", "ground_truth", "extra_info"])
os.makedirs(dataset_name, exist_ok=True)
df.to_json(os.path.join(dataset_name, "data.jsonl"), lines=True, orient="records")
