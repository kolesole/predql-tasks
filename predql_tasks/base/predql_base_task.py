"""Base PredQL task class."""

from abc import ABC, abstractmethod
from functools import lru_cache

from relbench.base import Dataset, TaskType
from relbench.metrics import (
##### REGRESSION metrics #####
    mae, 
    mse,
    r2,
##### BINARY CLASSIFICATION metrics #####
    roc_auc,
    average_precision,
    f1,
##### MULTICLASS CLASSIFICATION metrics #####
    accuracy,
    macro_f1,
    micro_f1,
##### MULTILABEL CLASSIFICATION metrics #####
    multilabel_auprc_macro,
    multilabel_auprc_micro,
    multilabel_f1_macro,
    multilabel_f1_micro,
##### LINK PREDICTION metrics #####
    link_prediction_recall,
    link_prediction_precision,
    link_prediction_map
)

from predql.base import Table
from predql.converter import Converter


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

    def compute_metrics(self, logits, labels):
        match self.task_type:
            case TaskType.REGRESSION:
                return {
                    "mae": mae(labels, logits),
                    "mse": mse(labels, logits),
                    "r2": r2(labels, logits)
                }
            case TaskType.BINARY_CLASSIFICATION:
                preds = (logits > 0.5).astype(int)
                return {
                    "roc_auc": roc_auc(labels, logits),
                    "average_precision": average_precision(labels, logits),
                    "f1": f1(labels, preds)
                }
            case TaskType.MULTICLASS_CLASSIFICATION:
                preds = logits.argmax(axis=1)
                return {
                    "accuracy": accuracy(labels, logits),
                    "macro_f1": macro_f1(labels, logits),
                    "micro_f1": micro_f1(labels, logits)
                }
            case TaskType.MULTILABEL_CLASSIFICATION:
                preds = (logits > 0.5).astype(int)
                return {
                    "auprc_macro": multilabel_auprc_macro(labels, logits),
                    "auprc_micro": multilabel_auprc_micro(labels, logits),
                    "f1_macro": multilabel_f1_macro(labels, preds),
                    "f1_micro": multilabel_f1_micro(labels, preds)
                }
            case TaskType.LINK_PREDICTION:
                return {
                    "recall": link_prediction_recall(labels, logits),
                    "precision": link_prediction_precision(labels, logits),
                    "map": link_prediction_map(labels, logits)
                }
            case _:
                pass

    @abstractmethod
    @lru_cache(maxsize=None)
    def _get_table(self, split: str) -> Table:
        r"""Get the task table for the given split.
        
        Must be implemented by the child class.

        Args:
            split (str): Split to get the task table for ("train"/"val"/"test").
        
        Returns:
            out (Table): Task table for the given split.
        """
        pass
