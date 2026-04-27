from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn
from relbench.base import TaskType
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType


class BaseModel(ABC, nn.Module):

    supported_task_types = [
        TaskType.BINARY_CLASSIFICATION,
        TaskType.REGRESSION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.MULTILABEL_CLASSIFICATION
    ]

    # to be set by subclasses
    gnn_name: str

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: dict,
        in_channels: int,
        gnn_config: dict,
        mlp_config: dict, # in_channels/hidden_channels/out_channels/layers/act/norm/bias
        task_type: TaskType,
        dropout: float=0.0,
    ) -> None:

        super().__init__()

        if task_type not in self.supported_task_types:
            raise ValueError(f"Unsupported task type: {task_type}")

        self.encoder = self._build_encoder(in_channels, data, col_stats_dict)
        self.temporal_encoder = self._build_temporal_encoder(in_channels, data)
        self.dropout = nn.Dropout(dropout)
        self.gnn = self._build_gnn(data, gnn_config)
        self.mlp_head = self._build_mlp_head(mlp_config)
        self.logits_func = self._build_logits_func(task_type)

        self.reset_parameters()

    @property
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self._reset_gnn_parameters()
        self.mlp_head.reset_parameters()

    def forward(self, batch: HeteroData, entity_table: NodeType) -> torch.Tensor:
        seed_time = batch[entity_table].seed_time
        seed_size = seed_time.size(0)
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self._apply_gnn(x_dict, batch)

        x = x_dict[entity_table][:seed_size]
        x = self.dropout(x)
        x = self.mlp_head(x)

        return self.logits_func(x)

    def _build_encoder(self, channels: int, data: HeteroData, col_stats_dict: dict) -> HeteroEncoder:
        return HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )

    def _build_temporal_encoder(self, channels: int, data: HeteroData) -> HeteroTemporalEncoder:
        return HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )

    def _build_mlp_head(self, mlp_config: dict) -> MLP:
        if not (in_channels := mlp_config.get("in_channels")):
            raise ValueError("MLP config must contain 'in_channels'")

        if not (out_channels := mlp_config.get("out_channels")):
            raise ValueError("MLP config must contain 'out_channels'")

        num_layers = mlp_config.get("layers", 1)
        if not (hidden_channels := mlp_config.get("hidden_channels")) and num_layers > 1:
            hidden_channels = in_channels

        return MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act=mlp_config.get("act", "relu"),
            norm=mlp_config.get("norm", "layer_norm"),
            dropout=mlp_config.get("dropout", 0.2),
            bias=mlp_config.get("bias", True),
        )

    def _build_logits_func(self, task_type: TaskType) -> Callable[[torch.Tensor], torch.Tensor]:
        match task_type:
            case TaskType.REGRESSION | TaskType.BINARY_CLASSIFICATION:
                return lambda x: x.squeeze(-1)
            case TaskType.MULTICLASS_CLASSIFICATION | TaskType.MULTILABEL_CLASSIFICATION:
                return lambda x: x
            case _:
                raise ValueError(f"Unsupported task type: {task_type}")

    @abstractmethod
    def _build_gnn(self, data: HeteroData, gnn_config: dict) -> nn.ModuleList:
        pass

    @abstractmethod
    def _apply_gnn(self, x_dict: dict, batch: HeteroData) -> dict:
        pass

    @abstractmethod
    def _reset_gnn_parameters(self) -> None:
        pass
