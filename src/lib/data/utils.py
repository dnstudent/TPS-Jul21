import os
from typing import List

import pandas as pd
import tensorflow as tf

features = ["deg_C", "relative_humidity", "absolute_humidity",
            "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5"]
targets = [f"target_{elem}" for elem in [
    "carbon_monoxide", "benzene", "nitrogen_oxides"]]


def train_data(data_path, delta=False, **kwargs):
    df = pd.read_csv(os.path.join(data_path, "train.csv"),
                     parse_dates=["date_time"],
                     index_col="date_time", **kwargs)
    if delta:
        df[targets] = df[targets].diff()
    return df.dropna()


def test_data(data_path, **kwargs):
    return pd.read_csv(os.path.join(data_path, "test.csv"),
                       parse_dates=["date_time"],
                       index_col="date_time", **kwargs)


def to_dataset(data):
    return tf.data.Dataset.from_tensor_slices(data)


def make_windowed_dataset(data, window_days, shift_hours, split_at=None):
    """Builds a windowed tf dataset from a pandas dataframe.
    """
    if split_at is not None:
        return make_windowed_dataset(data.iloc[:split_at], window_days, shift_hours).concatenate(
            make_windowed_dataset(data.iloc[split_at:], window_days, shift_hours))
    window_size = int(window_days * 24)
    return (
        to_dataset(data)
        .window(window_size, shift=shift_hours, drop_remainder=True)
        .flat_map(lambda x: x.batch(window_size, drop_remainder=True))
    )


def make_dates_dataset(data, window_days, shift_hours, offset_hours, as_dataset, split_at=None):
    """Builds a windowed tf dataset of input dates from a pandas dataframe's index. Dates are converted to hours since unix epoch.
    """
    if type(data) is pd.DataFrame:
        data = data.index
    if split_at is not None:
        a = make_dates_dataset(data[:split_at], window_days, shift_hours, offset_hours, as_dataset)
        b = make_dates_dataset(data[split_at:], window_days, shift_hours, offset_hours, as_dataset)
        if as_dataset:
            return a.concatenate(b)
        else:
            return a + b
    dates = (data[offset_hours:] - pd.to_datetime(0, unit="h", origin="unix")) // pd.Timedelta(1, unit="h")
    windowed_dataset = make_windowed_dataset(dates, window_days, shift_hours)
    if as_dataset:
        return windowed_dataset
    else:
        return list(map(lambda win_dates: pd.to_datetime(win_dates.tolist(), unit="h", origin="unix"), windowed_dataset.as_numpy_iterator()))


def make_supervised_dataset(data: pd.DataFrame | List[pd.DataFrame], features, targets, input_days, shift_hours, output_days, offset_hours, split_at=None) -> tf.data.Dataset:
    """Returns a zipped (input, output) windowed dataset.
    """
    if shift_hours is None:
        shift_hours = input_days * 24
    if output_days is None:
        output_days = input_days
    return tf.data.Dataset.zip((
        make_windowed_dataset(data[features], input_days, shift_hours, split_at),
        make_windowed_dataset(
            data[targets].iloc[offset_hours:], output_days, shift_hours, split_at)
    ))


def split_dataframe(data, validation_length, start, align_24h=True):
    if type(validation_length) is float:
        validation_length = int(len(data) * validation_length)
        if align_24h:
            validation_length -= (validation_length % 24)
    if type(start) is float:
        start = int(len(data) * start)
        if align_24h:
            start -= (start % 24)
    return (pd.concat((data.iloc[:start], data.iloc[start + validation_length:])), start), data.iloc[start:start + validation_length]
