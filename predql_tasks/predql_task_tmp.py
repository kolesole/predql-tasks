from functools import cached_property

import pandas as pd

from predql.base import Table
from predql.converter import TConverter
from predql_tasks.predql_base_task import PredQLBaseTask


class PredQLTaskTmp(PredQLBaseTask):
    time_col: str="timestamp"

    timedelta: pd.Timedelta
    num_eval_timestamps: int=1

    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp

    @cached_property
    def _train_timestamps(self) -> "pd.Series[pd.Timestamp]":
        start = self.val_timestamp - self.timedelta
        end = self.dataset.get_db(upto_test_timestamp=True).min_timestamp
        freq = -self.timedelta

        return pd.date_range(start=start, end=end, freq=freq)

    
    @cached_property
    def _val_timestamps(self) -> "pd.Series[pd.Timestamp]":
        start = self.val_timestamp
        end = min(
            self.val_timestamp
            + self.timedelta * (self.num_eval_timestamps - 1),
            self.test_timestamp - self.timedelta
        )
        freq = self.timedelta

        return pd.date_range(start=start, end=end, freq=freq)

    
    @cached_property
    def _test_timestamps(self) -> "pd.Series[pd.Timestamp]":
        start = self.test_timestamp
        end = min(
            self.test_timestamp
            + self.timedelta * (self.num_eval_timestamps - 1),
            self.dataset.get_db(upto_test_timestamp=False).max_timestamp - self.timedelta
        )
        freq = self.timedelta

        return pd.date_range(start=start, end=end, freq=freq)


    def _get_table(self,
                  split : str) -> Table:
        if self.converter is None:
            self.converter = TConverter(self.dataset.get_db(upto_test_timestamp=False), timestamps=None)

        if split == "train":
            timestamps = self._train_timestamps
        elif split == "val":
            timestamps = self._val_timestamps
        elif split == "test":
            timestamps = self._test_timestamps
        else:
            raise ValueError(f"Invalid split: {split}.")

        self.converter.set_timestamps(timestamps)

        return self.converter.convert(self.predql_query, execute=True)
