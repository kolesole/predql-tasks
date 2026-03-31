"""Base PredQL task class."""

from abc import ABC, abstractmethod
from functools import lru_cache

from relbench.base import Dataset, TaskType

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
            table.df.drop(columns=["label"], inplace=False)

        return table

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
