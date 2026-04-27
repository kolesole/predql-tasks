import torch.nn as nn
import torch.nn.functional as F
from relbench.base import TaskType
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
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

    def _build_gnn(self, data: HeteroData, gnn_config: dict) -> nn.ModuleDict:
        channels = gnn_config.get("channels", 128)
        layers = gnn_config.get("layers", 2)
        heads = gnn_config.get("heads", 4)

        gnn_layers = nn.ModuleList()
        norm_layers = nn.ModuleList()

        for _ in range(layers):
            gnn_layers.append(
                HGTConv(
                    in_channels=channels,
                    out_channels=channels,
                    metadata=data.metadata(),
                    heads=heads
                )
            )

            norm_dict = nn.ModuleDict({
                node_type: nn.LayerNorm(channels)
                for node_type in data.node_types
            })
            norm_layers.append(norm_dict)

        return nn.ModuleDict({
            "convs": gnn_layers,
            "norms": norm_layers
        })

    def _reset_gnn_parameters(self) -> None:
        for conv in self.gnn["convs"]:
            conv.reset_parameters()
        for norm_dict in self.gnn["norms"]:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def _apply_gnn(self, x_dict: dict, batch: HeteroData) -> dict:
        for i in range(len(self.gnn["convs"])):
            x_dict = self.gnn["convs"][i](x_dict, batch.edge_index_dict)

            new_x_dict = {}
            for node_type, x in x_dict.items():
                x = self.gnn["norms"][i][node_type](x)
                x = F.relu(x)
                x = self.dropout(x)
                new_x_dict[node_type] = x
            x_dict = new_x_dict

        return x_dict
