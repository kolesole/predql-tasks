"""Base PredQL task class."""

from abc import ABC, abstractmethod
from functools import cache

import numpy as np
from predql.base import Table
from predql.converter import Converter
from relbench.base import Dataset, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    macro_f1,
    mae,
    micro_f1,
    mse,
    multilabel_auprc_macro,
    multilabel_auprc_micro,
    multilabel_f1_macro,
    multilabel_f1_micro,
    r2,
    roc_auc,
)


class PredQLBaseTask(ABC):
    r"""Base PredQL task class with share attributes and methods.

    Attributes:
        converter (Converter): Converter to convert the PredQL query to a task table (Default=None).
        dataset (Dataset): Dataset to get the database for the task.
        predql_query (str): PredQL query to convert to a task table.
        task_type (TaskType): Task type of the task.
        entity_table (str): Name of the entity table in the database.
        entity_col (str): Name of the entity column in the task table (Default="fk").
        target_col (str): Name of the target column in the task table (Default="label").
    """
    # to be set by subclasses
    converter: Converter=None
    dataset: Dataset
    predql_query: str
    task_type: TaskType
    entity_table: str
    # for LINK_PREDICTION tasks
    dst_table: str=None
    # same for all tasks
    entity_col: str="fk"
    target_col: str="label"

    def get_table(self, split: str, hide_labels: bool=False) -> Table:
        r"""Get the task table for the given split.

        Args:
            split (str): Split to get the task table for ("train"/"val"/"test").
            hide_labels (bool): Whether to hide the labels in the task table (Default=False).

        Returns:
            out (Table): Task table for the given split.
        """
        table = self._get_table(split)

        if hide_labels:
            table.df.drop(columns=["label"], inplace=True)

        return table

    def compute_metrics(self, logits: np.ndarray, labels: np.ndarray) -> dict:
        r"""Compute the metrics for the given logits and labels.

        Args:
            logits (np.ndarray): Logits predicted by the model.
            labels (np.ndarray): True labels.

        Returns:
            out (dict): Dictionary of metric names and their values.
        """
        match self.task_type:
            case TaskType.REGRESSION:
                return {
                    "mae": mae(labels, logits),
                    "mse": mse(labels, logits),
                    "r2": r2(labels, logits)
                }
            case TaskType.BINARY_CLASSIFICATION:
                return {
                    "accuracy": accuracy(labels, logits),
                    "roc_auc": roc_auc(labels, logits),
                    "average_precision": average_precision(labels, logits),
                    "f1": f1(labels, logits)
                }
            case TaskType.MULTICLASS_CLASSIFICATION:
                return {
                    "accuracy": accuracy(labels, logits),
                    "macro_f1": macro_f1(labels, logits),
                    "micro_f1": micro_f1(labels, logits)
                }
            case TaskType.MULTILABEL_CLASSIFICATION:
                return {
                    "auprc_macro": multilabel_auprc_macro(labels, logits),
                    "auprc_micro": multilabel_auprc_micro(labels, logits),
                    "f1_macro": multilabel_f1_macro(labels, logits),
                    "f1_micro": multilabel_f1_micro(labels, logits)
                }
            case TaskType.LINK_PREDICTION:
                return {
                    "roc_auc": roc_auc(labels, logits),
                    "average_precision": average_precision(labels, logits),
                    "f1": f1(labels, logits)
                }
            case _:
                pass

    @abstractmethod
    @cache
    def _get_table(self, split: str) -> Table:
        r"""Get the task table for the given split.

        Must be implemented by the child class.

        Args:
            split (str): Split to get the task table for ("train"/"val"/"test").

        Returns:
            out (Table): Task table for the given split.
        """
        pass
