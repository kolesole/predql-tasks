from relbench.base import TaskType

import torch.nn as nn

from torch_geometric.data import HeteroData
from torch_frame.data.stats import StatType
from torch_geometric.typing import NodeType
from torch_geometric.nn import HGTConv

from .base_model import BaseModel


class HGTModel(BaseModel):

    gnn_name: str="HGT"

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: dict[str, dict[str, dict[StatType, any]]],
        in_channels: int,
        gnn_config: dict, # channels/heads/layers
        mlp_config: dict,
        task_type: TaskType,
        dropout: float = 0.3,
        shallow_list: list[NodeType] = [],
        id_awareness: bool = False,
        is_temporal: bool=False
    ):
        super().__init__(
            data, 
            col_stats_dict, 
            in_channels, 
            gnn_config,
            mlp_config, 
            task_type, 
            dropout, 
            shallow_list, 
            id_awareness, 
            is_temporal
        )
    
    def _build_gnn(self, data: HeteroData, gnn_config: dict) -> nn.ModuleList:
        return nn.ModuleList([
            HGTConv(
                in_channels=gnn_config.get("channels", 128),
                out_channels=gnn_config.get("channels", 128),
                metadata=data.metadata(),
                heads=gnn_config.get("heads", 4)
            ) for _ in range(gnn_config.get("layers", 2))
        ])
    
    def _reset_gnn_parameters(self):
        for conv in self.gnn:
            conv.reset_parameters()
    
    def _apply_gnn(self, x_dict: dict, batch: HeteroData) -> dict:
        for conv in self.gnn:
            x_dict = conv(x_dict, batch.edge_index_dict)
        return x_dict
    