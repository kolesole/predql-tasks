from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.nn import MLP

from relbench.base import TaskType
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder


class BaseModel(ABC, nn.Module):

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
        shallow_list: list[NodeType]=[],
        id_awareness: bool=False,
        is_temporal: bool=False
    ) -> None:
    
        super().__init__()
        
        self.encoder = self._build_encoder(in_channels, data, col_stats_dict)
        self.temporal_encoder = (
            self._build_temporal_encoder(in_channels, data) 
            if is_temporal else None
        )

        self.embedding_dict = self._build_embedding_dict(in_channels, data, shallow_list)
        self.id_awareness_emb = (
            nn.Embedding(1, in_channels) 
            if id_awareness else None
        )

        self.gnn = self._build_gnn(data, gnn_config)
        self.dropout = nn.Dropout(dropout)
        self.mlp_head = self._build_mlp_head(mlp_config)
        self.logits_func = self._build_logits_func(task_type)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        if self.temporal_encoder:
            self.temporal_encoder.reset_parameters()

        for emb in self.embedding_dict.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.1)

        if self.id_awareness_emb:
            nn.init.xavier_uniform_(self.id_awareness_emb.weight)

        self._reset_gnn_parameters()

        self.mlp_head.reset_parameters()
    
    def forward(self, batch: HeteroData, entity_table: NodeType) -> torch.Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        if self.temporal_encoder:
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self._apply_gnn(x_dict, batch)

        x = x_dict[entity_table][: seed_time.size(0)]
        x = self.dropout(x)
        x = self.mlp_head(x)

        return self.logits_func(x)

    def forward_dst_readout(self, batch: HeteroData, entity_table: NodeType, dst_table: NodeType) -> torch.Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        if self.temporal_encoder:
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self._apply_gnn(x_dict, batch)

        x = x_dict[dst_table]
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

    def _build_embedding_dict(self, channels: int, data: HeteroData, shallow_list: list[NodeType]) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                node: nn.Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )
    
    def _build_mlp_head(self, mlp_config):
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
            norm=mlp_config.get("norm", "batch_norm"),
            bias=mlp_config.get("bias", True),
        )
    
    def _build_logits_func(self, task_type):
        match task_type:
            case TaskType.REGRESSION | TaskType.BINARY_CLASSIFICATION | TaskType.LINK_PREDICTION:
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
    def _reset_gnn_parameters(self):
        pass
