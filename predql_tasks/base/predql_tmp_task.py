"""Temporal PredQL task class."""

from functools import cached_property

import pandas as pd

from predql.base import Table
from predql.converter import TConverter

from .predql_base_task import PredQLBaseTask


class PredQLTmpTask(PredQLBaseTask):
    r"""Temporal PredQL task class.
    
    This class is used for temporal tasks with time predictions.

    Attributes:
        timedelta (str): Time delta for the time predictions.
        num_eval_timestamps (int): Number of evaluation timestamps for the validation and test splits (Default=1).
        val_timestamp (str): Validation timestamp for the time predictions.
        test_timestamp (str): Test timestamp for the time predictions.
        time_col (str): Name of the time column in the task table (Default="timestamp").
    """
    # to be set by subclasses
    timedelta: pd.Timedelta
    num_eval_timestamps: int=1
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp
    # same for all tasks
    time_col: str="timestamp"

    @cached_property
    def _train_timestamps(self) -> "pd.Series[pd.Timestamp]":
        r"""Get the training timestamps for the time predictions.
        
        Returns:
            out (pd.Series[pd.Timestamp]): Training timestamps for the time predictions.
        """
        start = self.val_timestamp - self.timedelta
        end = self.dataset.get_db(upto_test_timestamp=True).min_timestamp
        freq = -self.timedelta

        return pd.date_range(start=start, end=end, freq=freq)

    
    @cached_property
    def _val_timestamps(self) -> "pd.Series[pd.Timestamp]":
        r"""Get the validation timestamps for the time predictions.

        Returns:
            out (pd.Series[pd.Timestamp]): Validation timestamps for the time predictions.
        """
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
        r"""Get the test timestamps for the time predictions.

        Returns:
            out (pd.Series[pd.Timestamp]): Test timestamps for the time predictions.
        """
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
        r"""Get the task table for the given split.

        Args:
            split (str): Split to get the task table for ("train"/"val"/"test").
        
        Returns:
            out (Table): Task table for the given split.
        """
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
