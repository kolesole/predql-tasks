from relbench.base import TaskType

import torch.nn as nn

from torch_geometric.data import HeteroData
from torch_frame.data.stats import StatType
from torch_geometric.typing import NodeType
from torch_geometric.nn import HeteroConv, GINConv, MLP

from .base_model import BaseModel


class GINModel(BaseModel):

    gnn_name: str="GIN"

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: dict[str, dict[str, dict[StatType, any]]],
        in_channels: int,
        gnn_config: dict, # channels/mlp_[layers|dropout|act|norm|bias]/train_eps/layers
        mlp_config: dict,
        task_type: TaskType,
        dropout: float=0.3,
        shallow_list: list[NodeType]=[],
        id_awareness: bool=False,
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
        channels = gnn_config.get("channels", 128)

        return nn.ModuleList([
            HeteroConv({
                edge_type: GINConv(
                    MLP(
                        in_channels=channels,
                        hidden_channels=channels,
                        out_channels=channels,
                        num_layers=gnn_config.get("mlp_layers", 2),
                        dropout=gnn_config.get("mlp_dropout", 0.3),
                        act=gnn_config.get("mlp_act", "relu"),
                        norm=gnn_config.get("mlp_norm", "batch_norm"),
                        bias=gnn_config.get("mlp_bias", True)
                    ),
                    train_eps=gnn_config.get("train_eps", True),
                ) for edge_type in data.edge_types
            }) for _ in range(gnn_config.get("layers", 2))
        ])
    
    def _reset_gnn_parameters(self):
        for conv in self.gnn:
            conv.reset_parameters()
    
    def _apply_gnn(self, x_dict: dict, batch: HeteroData) -> dict:
        for conv in self.gnn:
            x_dict = conv(x_dict, batch.edge_index_dict)
        return x_dict
    