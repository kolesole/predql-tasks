from abc import ABC, abstractmethod
from functools import lru_cache

from relbench.base import Dataset, TaskType

from predql.base import Table
from predql.converter import Converter


class PredQLBaseTask(ABC):
    converter: Converter=None
    dataset: Dataset
    predql_query: str
    task_type: TaskType

    entity_table: str
    entity_col: str="fk"
    target_col: str="label"

    def get_table(self,
                  split : str,
                  hide_labels : bool=False) -> Table:
        table = self._get_table(split)

        if hide_labels:
            table.df.drop(columns=["label"], inplace=False)

        return table


    @abstractmethod
    @lru_cache(maxsize=None)
    def _get_table(self, split: str) -> Table:
        pass
