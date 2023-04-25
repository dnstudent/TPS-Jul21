from typing import List

import tensorflow as tf
import keras
import pandas as pd
from ..data.windowed_data import WTSMaker


class PollutionEstimator(keras.Sequential):
    """Assumes that the input is contiguous hourly data
    """

    def __init__(self,
                 features, targets, pollution_model, targets_normalization, compilation_kwargs=None,
                 *args, **kwargs):
        """Initializes a pollution model.

        Args:
            features (List[str]): The list of features to use as input
            targets (List[str]): The list of targets to predict
            input_days (int | float): The number of days worth of data to use as input (=> temporal size of the input window)
            output_days (int | float): The number of days worth of data to predict (=> temporal size of the output window)
            offset_days (int | float): The offset between the beginning of the input and the beginning of the output in days
        """
        super().__init__(*args, **kwargs)
        if compilation_kwargs is None:
            compilation_kwargs = {"loss": "mse"}
        self.dataset_maker = WTSMaker(
            features, targets, pollution_model.input_days, pollution_model.output_days, pollution_model.offset_hours)
        self.compilation_kwargs = compilation_kwargs
        self.optimizer_ser = self.compilation_kwargs.pop("optimizer", {"class_name": "sgd", "config": {}})
        self.features = features
        self.targets = targets
        self.prediction_shift = self.dataset_maker.output_size

        # Features normalization layers: (batch_size, input_size, n_features) => (batch_size, input_size, n_features)
        self.features_normalization = keras.layers.Normalization(axis=-1, name=f"{self.name}_features_normalization")
        self.add(self.features_normalization)

        # Pollution model: (batch_size, input_size, n_features) => (batch_size, ?, n_targets)
        self.pollution_model = pollution_model
        self.pollution_model.build(input_shape=(
            None, self.dataset_maker.input_size, len(features)))
        self.add(self.pollution_model)

        # Reshape: (batch_size, ?, n_targets) => (batch_size, output_size, n_targets)
        self.reshape = keras.layers.Reshape(
            (self.dataset_maker.output_size, len(targets)), name=f"{self.name}_reshape")
        self.add(self.reshape)

        # Targets reconstruction: (batch_size, output_size, n_targets) => (batch_size, output_size, n_targets)
        self.targets_reconstruction = keras.layers.Normalization(
            axis=-1, invert=True, name=f"{self.name}_targets_reconstruction")
        self.add(self.targets_reconstruction)

        self.build(input_shape=(
            None, self.dataset_maker.input_size, len(features)))

        if targets_normalization:
            self.targets_normalization = keras.layers.Normalization(
                axis=-1, name=f"{self.name}_targets_normalization")
        else:
            self.targets_normalization = None

    def adapt_training(self, training_data):
        """Adapts the normalization layers to the TRAINING data. One MUST assure that the data is already batched,
        if training_data is a tf dataset.
        """
        self.features_normalization.reset_state()
        self.targets_reconstruction.reset_state()
        if isinstance(training_data, (list, tuple)):
            training_data = pd.concat(training_data)
        if type(training_data) is pd.DataFrame:
            self.features_normalization.adapt(
                training_data[self.features].to_numpy().reshape((1, -1, len(self.features))))
            self.targets_reconstruction.adapt(
                training_data[self.targets].to_numpy().reshape((1, -1, len(self.targets))))
            if self.targets_normalization:
                self.targets_normalization.adapt(
                    training_data[self.targets].to_numpy().reshape((1, -1, len(self.targets))))
        else:
            if len(training_data.element_spec[0].shape) != 3 or len(training_data.element_spec[1].shape) != 3:
                raise ValueError(
                    "Training data must be batched (i.e. have shape (batch_size, input_size, n_features) and ("
                    "batch_size, output_size, n_targets))")
            self.features_normalization.adapt(
                training_data.map(lambda x, _: x))
            self.targets_reconstruction.adapt(
                training_data.map(lambda _, y: y))
            if self.targets_normalization:
                self.targets_normalization.adapt(
                    training_data.map(lambda _, y: y))

    def trim_to_forecast(self, data):
        """Trims the data to the appropriate size for the model.
        """
        rem = ((len(data) - self.dataset_maker.input_size) %
               self.prediction_shift)
        if rem != 0:
            return data.iloc[:-rem]
        return data

    def train(self, data: List[pd.DataFrame] | pd.DataFrame, epochs, shift_hours, validation_data=None, batch_size=32, cache_to_disk=True, **kwargs):
        """Trains the model on a dataset, assuming the input is in my format.
        """
        def cache_path(train):
            if not cache_to_disk:
                return None
            import os
            if not os.path.exists(".cache"):
                os.mkdir(".cache")
            return f".cache/{self.name}_{self.dataset_maker.input_size}_{'train' if train else 'validation'}"

        training_dataset = self.dataset_maker.supervised_dataset(
            data, shift_hours).cache(cache_path(True)).shuffle(10_000).batch(batch_size)
        if validation_data is not None:
            validation_data = self.dataset_maker.supervised_dataset(
                validation_data, self.dataset_maker.output_size).batch(batch_size).cache(cache_path(False))
        return super().fit(training_dataset.prefetch(tf.data.AUTOTUNE), validation_data=validation_data.prefetch(tf.data.AUTOTUNE), epochs=epochs, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        """Trains the model on a dataset, assuming the input is in keras/scikit format.
        """
        if isinstance(X, tf.data.Dataset):
            return super().fit(X, y, *args, **kwargs)
        try:
            if isinstance(X, (list, tuple)):
                data = [pd.concat([xx, yy], axis=1) for xx, yy in zip(X, y)]
            else:
                data = pd.concat([X, y], axis=1)
            return self.train(data, shift_hours=kwargs.pop("shift_hours", 1), validation_data=kwargs.pop("validation_data", None), *args, **kwargs)
        except TypeError as e:
            print(e)
            return self.train(X, y, *args, **kwargs)
        except RuntimeError as e:
            print(e)
            return super().fit(X, y, *args, **kwargs)

    def forecast(self, data, batch_size=32):
        """Returns the predictions of the model on a dataset.
        """
        # data = self.trim_to_forecast(data)
        # Assumes output windows are contiguous and non-overlapping
        raw_results = self.predict(
            self.dataset_maker.input_dataset(data, shift_hours=self.dataset_maker.output_size).batch(batch_size))
        dates = self.dataset_maker.continuous_prediction_dates(data)
        return pd.DataFrame(raw_results.reshape((-1, len(self.targets))), index=dates, columns=self.targets).rename_axis(index="date_time")

    def assess(self, data, metrics, batch_size=32):
        """Returns the metrics of the model on a dataset.
        """
        # Assumes windows are contiguous and non-overlapping
        return self.evaluate(
            self.dataset_maker.supervised_dataset(data, shift_hours=self.dataset_maker.output_size).batch(batch_size), return_dict=True)

    # def compile(self):
    #     super().compile(optimizer=keras.optimizers.deserialize(self.optimizer_ser), **self.compilation_kwargs)
        # super().compile(**self.compilation_kwargs)
