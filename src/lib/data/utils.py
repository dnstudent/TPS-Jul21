import os

import pandas as pd
import tensorflow as tf

features = ["deg_C", "relative_humidity", "absolute_humidity",
            "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5"]
targets = [f"target_{elem}" for elem in [
    "carbon_monoxide", "benzene", "nitrogen_oxides"]]


def manage_pair(func):
    def wrapper(data, *args, **kwargs):
        if type(data) is list or type(data) is tuple:
            return func(data[0], *args, **kwargs).concatenate(func(data[1], *args, **kwargs))
        return func(data, *args, **kwargs)
    return wrapper


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


def to_dataset(data: pd.DataFrame):
    return tf.data.Dataset.from_tensor_slices(data)


@manage_pair
def make_windowed_dataset(data, window_size, shift_size, subset=None, offset=None):
    """Builds a windowed tf dataset from a pandas dataframe.
    """
    if subset is not None:
        data = data[subset]
    if offset is not None:
        data = data.iloc[offset:]
    return (
        to_dataset(data)
        .window(window_size, shift=shift_size, drop_remainder=True)
        .flat_map(lambda x: x.batch(window_size, drop_remainder=True))
    )


def windowed_dates(data, window_size, shift_size, skip, as_dataset):
    """Builds a windowed tf dataset of input dates from a pandas dataframe's index. Dates are converted to hours since unix epoch.
    """
    @manage_pair
    def make_windowed_dates(data):
        if isinstance(data, pd.DataFrame):
            data = data.index
        dates = (data[skip:] - pd.to_datetime(0, unit="h", origin="unix")) // pd.Timedelta(1, unit="h")
        return make_windowed_dataset(dates, window_size, shift_size, None, None)
    windowed_dataset = make_windowed_dates(data)
    if as_dataset:
        return windowed_dataset
    else:
        return list(map(lambda win_dates: pd.to_datetime(win_dates.tolist(), unit="h", origin="unix"), windowed_dataset.as_numpy_iterator()))


@manage_pair
def make_supervised_dataset(data: pd.DataFrame, features, targets, input_size, shift_size, output_size, offset_size) -> tf.data.Dataset:
    """Returns a zipped (input, output) windowed dataset.
    """
    return tf.data.Dataset.zip((
        make_windowed_dataset(
            data, input_size, shift_size, subset=features),
        make_windowed_dataset(
            data, output_size, shift_size, subset=targets, offset=offset_size)
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
    return [data.iloc[:start].copy(), data.iloc[start + validation_length:].copy()], data.iloc[start:start + validation_length].copy()


def split_as_sklearn(data):
    if isinstance(data, list) or isinstance(data, tuple):
        tr = [split_as_sklearn(elem) for elem in data]
        return [elem[0] for elem in tr], [elem[1] for elem in tr]
    return data[features], data[targets]
