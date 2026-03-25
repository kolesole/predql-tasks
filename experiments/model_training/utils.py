from  typing import Callable

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from predql.base import Table
from tasks.predql_base_task import PredQLBaseTask


def get_transform(table: Table, task: PredQLBaseTask) -> "Callable[[HeteroData], HeteroData]":
    target_map = torch.from_numpy(table.df["label"].values)

    if task.task_type in ["binary_classification", "multiclass_classification"]:
        target_map = target_map.to(torch.long)
    else:
        target_map = target_map.to(torch.float)
    
    def transform(batch):
        batch.y = target_map[batch[task.entity_table].input_id]
        return batch

    return transform


def get_loader_dict(
        task: PredQLBaseTask, 
        data: HeteroData, device: torch.device):
    loader_dict = {}

    for split in ["train", "val", "test"]:
        table = task.get_table(split)

        times = table.df[task.time_col].values.astype('datetime64[s]').astype('int64')
        
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[128, 128],
            input_nodes=(
                task.entity_table,
                torch.from_numpy(table.df[task.entity_col].values).to(torch.long)
            ),
            input_time=torch.from_numpy(times).to(torch.long),
            time_attr="time",
            transform=get_transform(table, task),
            batch_size=512,
            temporal_strategy="uniform",
            shuffle=split == "train",
            num_workers=0,
            persistent_workers=False
        )