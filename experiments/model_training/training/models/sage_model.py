import torch.nn as nn
from relbench.base import TaskType
from relbench.modeling.nn import HeteroGraphSAGE
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData

from .base_model import BaseModel


class SAGEModel(BaseModel):

    gnn_name: str="SAGE"

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: dict[str, dict[str, dict[StatType, any]]],
        in_channels: int,
        gnn_config: dict, # channels/aggr/layers
        mlp_config: dict,
        task_type: TaskType,
        dropout: float = 0.3
    ) -> None:
        super().__init__(
            data,
            col_stats_dict,
            in_channels,
            gnn_config,
            mlp_config,
            task_type,
            dropout
        )

    def _build_gnn(self, data: HeteroData, gnn_config: dict) -> nn.ModuleList:
        return HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=gnn_config.get("channels", 128),
            aggr=gnn_config.get("aggr", "mean"),
            num_layers=gnn_config.get("layers", 2)
        )

    def _reset_gnn_parameters(self) -> None:
        self.gnn.reset_parameters()

    def _apply_gnn(self, x_dict: dict, batch: HeteroData) -> dict:
        return self.gnn(x_dict, batch.edge_index_dict)
