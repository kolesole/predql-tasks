import os
from getpass import getpass

import numpy as np
import pandas as pd
import torch
import yaml
from predql.base import Table
from relbench.base import TaskType
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from predql_tasks.base import PredQLBaseTask


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    return device


def set_hf_token(token: str=None) -> None:
    if not token:
        # chekc if the token file exists
        if os.path.exists("hf_token.txt"):
            with open("hf_token.txt") as f:
                token = f.read().strip()

        if not token:
            token = getpass("Enter your Hugging Face token: ").strip()

    os.environ["HF_TOKEN"] = token


def load_config(config_path: str="config.yml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File does not exist: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def patched_to_unix_time(ser: pd.Series) -> np.ndarray:
    unix_time = ser.astype("int64").values.copy()
    if ser.dtype == np.dtype("datetime64[ns]"):
        unix_time //= 10**9
    return unix_time


def get_transform(table, task: PredQLBaseTask):
    labels_list = table.df["label"].tolist()
    labels_stacked = np.array(labels_list)
    target_map = torch.from_numpy(labels_stacked)

    if task.task_type in [TaskType.BINARY_CLASSIFICATION]:
        target_map = target_map.to(torch.long)
    else:
        target_map = target_map.to(torch.float)

    def transform(batch):
        batch.y = target_map[batch[task.entity_table].input_id]
        return batch

    return transform


def encode_labels(
    task: PredQLBaseTask,
    binarize: bool=False,
    encode: bool=False
) -> tuple[dict[str, Table], MultiLabelBinarizer | None, LabelEncoder | None]:
    table_dict = {}
    mlb, le = None, None

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    all_labels = (
        list(train_table.df["label"]) +
        list(val_table.df["label"]) +
        list(test_table.df["label"])
    )

    if binarize:
        mlb = MultiLabelBinarizer()
        mlb.fit(all_labels)

        train_table.df["label"] = list(mlb.transform(train_table.df["label"]))
        val_table.df["label"] = list(mlb.transform(val_table.df["label"]))
        test_table.df["label"] = list(mlb.transform(test_table.df["label"]))

    if encode:
        le = LabelEncoder()
        le.fit(all_labels)

        train_table.df["label"] = le.transform(train_table.df["label"])
        val_table.df["label"] = le.transform(val_table.df["label"])
        test_table.df["label"] = le.transform(test_table.df["label"])

    table_dict = {
        "train": train_table,
        "val": val_table,
        "test": test_table
    }

    return table_dict, mlb, le


def make_loaders(
    data: HeteroData,
    task: PredQLBaseTask,
    batch_size: int,
    num_neighbors: list[int],
    binarize: bool=False,
    encode: bool=False
) -> tuple[dict[str, NeighborLoader], MultiLabelBinarizer | None, LabelEncoder | None]:
    loader_dict = {}

    table_dict, mlb, le = encode_labels(task, binarize, encode)

    for split, table in table_dict.items():
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

    return loader_dict, mlb, le


def compute_pos_weight(task: PredQLBaseTask) -> float:
    if task.task_type != TaskType.BINARY_CLASSIFICATION:
        raise ValueError("Pos weights can only be computed for binary classification tasks.")

    train_table = task.get_table("train")
    class_counts = train_table.df["label"].value_counts()

    neg_count = class_counts.get(0, 0)
    pos_count = class_counts.get(1, 0)

    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"Class weights computed: pos_weight={pos_weight:.2f}")
    return pos_weight
