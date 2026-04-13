import os

import redelex
from relbench.base import TaskType

from getpass import getpass

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

from predql_tasks.base import PredQLBaseTask


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    return device


def set_hf_token(token=None):
    if not token:
        # chekc if the token file exists
        if os.path.exists("hf_token.txt"):
            with open("hf_token.txt", "r") as f:
                token = f.read().strip()
        
        if not token:
            token = getpass("Enter your Hugging Face token: ").strip()
            
    os.environ["HF_TOKEN"] = token


def patched_to_unix_time(ser):
    unix_time = ser.astype("int64").values.copy()
    if ser.dtype == np.dtype("datetime64[ns]"):
        unix_time //= 10**9
    return unix_time


def get_transform(table, task):
    target_map = torch.from_numpy(table.df["label"].values)

    if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
        target_map = target_map.to(torch.long)
    else:
        target_map = target_map.to(torch.float)
    
    def transform(batch):
        batch.y = target_map[batch[task.entity_table].input_id]
        return batch

    return transform


def make_loaders(data, task, batch_size, num_neighbors, is_temporal=False):
    loader_dict = {}

    for split in ["train", "val", "test"]:
        table = task.get_table(split)

        input_time = time_attr = None
        if is_temporal:
            times = table.df[task.time_col].values.astype('datetime64[s]').astype('int64')
            input_time = torch.from_numpy(times).to(torch.long)
            time_attr = "time"
        
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            input_nodes=(
                task.entity_table,
                torch.from_numpy(table.df[task.entity_col].values).to(torch.long)
            ),
            input_time=input_time,
            time_attr=time_attr,
            transform=get_transform(table, task),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
            persistent_workers=False
        )
    
    return loader_dict


def compute_pos_weight(task: PredQLBaseTask):
    if task.task_type != TaskType.BINARY_CLASSIFICATION:
        raise ValueError("Pos weights can only be computed for binary classification tasks.")
    
    train_table = task.get_table("train")
    class_counts = train_table.df["label"].value_counts()
    if len(class_counts) > 1:
        neg_count = class_counts.iloc[0]  
        pos_count = class_counts.iloc[1] 
        pos_weight = neg_count / pos_count
    else:
        pos_weight = 1.0
    print(f"Class weights computed: pos_weight={pos_weight:.2f}")

    return pos_weight
