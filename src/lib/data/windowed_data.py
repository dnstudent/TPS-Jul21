from typing import List

import pandas as pd

from .utils import make_windowed_dataset, make_supervised_dataset, windowed_dates


class WTSMaker:
    """A windowed dataset of dates from a pandas dataframe's index. Assumes that each entry is an hour apart."""

    def __init__(self, features, targets, input_days, output_days, offset_hours):
        self.features = features
        self.targets = targets
        self.input_days = input_days
        self.input_size = round(input_days * 24)
        self.output_days = output_days
        self.output_size = round(output_days * 24)
        self.offset_size = offset_hours

    # Data windowing
    def input_dataset(self, data: pd.DataFrame | List[pd.DataFrame], shift_hours):
        """Builds a windowed tf dataset from a pandas dataframe.
        """
        return make_windowed_dataset(data, self.input_size, shift_hours, subset=self.features)

    def output_dataset(self, data: pd.DataFrame | List[pd.DataFrame], shift_hours):
        """Builds a windowed tf dataset from a pandas dataframe.
        """
        return make_windowed_dataset(data, self.output_days, shift_hours, subset=self.targets, offset=self.offset_size)

    def supervised_dataset(self, data: pd.DataFrame, shift_hours):
        """Returns a zipped (input, output) windowed dataset.
        """
        return make_supervised_dataset(data, self.features, self.targets, self.input_size, shift_hours,
                                       self.output_size, self.offset_size)

    # Dates windowing
    def _dates(self, data, shift_hours, as_dataset):
        """Builds a windowed tf dataset of input dates from a pandas dataframe's index. Dates are converted to hours
        since unix epoch.
        """
        return windowed_dates(data, self.input_size, shift_hours, self.offset_size, as_dataset)

    def input_dates(self, data: pd.DataFrame, shift_hours, as_dataset=False):
        """Builds a windowed tf dataset of input dates from a pandas dataframe's index. Dates are converted to hours
        since unix epoch.
        """
        return self._dates(data, 0, shift_hours, as_dataset)

    def output_dates(self, data: pd.DataFrame, shift_hours, as_dataset=False):
        """Builds a windowed tf dataset of output  dates from a pandas dataframe's index. Dates are converted to
        hours since unix epoch.
        """
        return self._dates(data, self.offset_size, shift_hours, as_dataset)

    def continuous_prediction_dates(self, data):
        """Returns a pandas series of dates corresponding to the prediction dates. Expects a contiguous hourly
        measures dataframe.
        """
        # Computing the number of windows in data assuming a window size of self.input_size
        if isinstance(data, list) or isinstance(data, tuple):
            return pd.concat([self.continuous_prediction_dates(d) for d in data])
        n_windows = (len(data) - self.input_size) // self.output_size + 1
        start = data.index[0] + pd.Timedelta(self.offset_size, unit="h")
        return pd.date_range(start, start + (n_windows * self.output_size - 1) * pd.Timedelta("1h"), freq="h")
