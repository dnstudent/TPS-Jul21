import pandas as pd

from .utils import make_windowed_dataset, make_supervised_dataset, make_dates_dataset


class WTSMaker:
    """A windowed dataset of dates from a pandas dataframe's index. Assumes that each entry is an hour apart."""

    def __init__(self, features, targets, input_days, output_days, offset_hours, shift_hours):
        self.features = features
        self.targets = targets
        self.input_days = input_days
        self.input_size = int(input_days * 24)
        self.output_days = output_days
        self.output_size = int(output_days * 24)
        self.offset_size = offset_hours
        self.shift_size = shift_hours

    # Data windowing
    def input_dataset(self, data: pd.DataFrame, split_at=None):
        """Builds a windowed tf dataset from a pandas dataframe.
        """
        return make_windowed_dataset(data[self.features], self.input_days, self.shift_size, split_at)

    def output_dataset(self, data: pd.DataFrame, split_at=None):
        """Builds a windowed tf dataset from a pandas dataframe.
        """
        return make_windowed_dataset(data[self.targets].iloc[self.offset_size:], self.output_days, self.shift_size, split_at)

    def supervised_dataset(self, data: pd.DataFrame, split_at=None):
        """Returns a zipped (input, output) windowed dataset.
        """
        return make_supervised_dataset(data, self.features, self.targets, self.input_days, self.shift_size, self.output_days, self.offset_size, split_at)

    # Dates windowing
    def _dates(self, data, offset, as_dataset, split_at):
        """Builds a windowed tf dataset of input dates from a pandas dataframe's index. Dates are converted to hours since unix epoch.
        """
        return make_dates_dataset(data, self.input_days, self.shift_size, offset, as_dataset, split_at)

    def input_dates(self, data: pd.DataFrame, shift_size=1, as_dataset=False, split_at=None):
        """Builds a windowed tf dataset of input dates from a pandas dataframe's index. Dates are converted to hours since unix epoch.
        """
        return self._dates(data, 0, as_dataset, split_at)

    def output_dates(self, data: pd.DataFrame, shift_size=1, as_dataset=False, split_at=None):
        """Builds a windowed tf dataset of output  dates from a pandas dataframe's index. Dates are converted to hours since unix epoch.
        """
        return self._dates(data, self.offset_size, as_dataset, split_at)

    def continuous_prediction_dates(self, data):
        """Returns a pandas series of dates corresponding to the prediction dates. Expects a contiguous hourly measures dataframe.
        """
        # Computing the number of windows in data assuming a window size of self.input_wsize and a shift between windows of self.prediction_shift
        n_windows = (len(data) - self.input_size) // self.output_size + 1
        start = data.index[0] + pd.Timedelta(self.offset_size, unit="h")
        return pd.date_range(start, start + (n_windows * self.output_size - 1) * pd.Timedelta("1h"), freq="h")

    # def size(self, data):
    #     """Returns the number of windows in a dataset."""
    #     return min(len(data) - self.input_size + self.shift_size, len(data) - self.output_size + self.shift_size)
