import pandas as pd
import keras

from ..data.utils import targets
from ..data.windowed_data import WTSMaker


class PollutionEstimator(keras.Sequential):
    """Assumes that the input is contiguous hourly data
    """

    def __init__(self, features, targets, input_days, output_days, offset_hours, shift_hours, pollution_model, compilation_kwargs={"loss": "mse"}, *args, **kwargs):
        """Initializes a pollution model.

        Args:
            features (List[str]): The list of features to use as input
            targets (List[str]): The list of targets to predict
            input_days (int | float): The number of days worth of data to use as input (=> temporal size of the input window)
            output_days (int | float): The number of days worth of data to predict (=> temporal size of the output window)
            offset_days (int | float): The offset between the beginning of the input and the beginning of the output in days
        """
        super().__init__(*args, **kwargs)
        self.dataset_maker = WTSMaker(features, targets, input_days, output_days, offset_hours, shift_hours)
        self.compilation_kwargs = compilation_kwargs
        self.features = features
        self.targets = targets
        self.prediction_shift = self.dataset_maker.output_size
        self.features_normalization = keras.layers.Normalization(axis=-1)
        self.add(self.features_normalization)
        self.pollution_model = pollution_model
        self.pollution_model.build(input_shape=(None, self.dataset_maker.input_size, len(features)))
        self.add(self.pollution_model)
        self.reshape = keras.layers.Reshape((self.dataset_maker.output_size, len(targets)))
        self.add(self.reshape)
        # self.targets_normalization = keras.layers.Normalization(axis=-1)
        self.targets_reconstruction = keras.layers.Normalization(axis=-1, invert=True)
        self.add(self.targets_reconstruction)
        self.build(input_shape=(None, self.dataset_maker.input_size, len(features)))

    # def call(self, inputs, *args, **kwargs):
    #     x = self.features_normalization(inputs, *args, **kwargs)
    #     x = self.pollution_model.call(x, *args, **kwargs)
    #     return self.targets_reconstruction(x, *args, **kwargs)

    def adapt_training(self, training_data):
        """Adapts the normalization layers to the TRAINING data.
        """
        self.features_normalization.reset_state()
        self.targets_reconstruction.reset_state()
        if type(training_data) is pd.DataFrame:
            self.features_normalization.adapt(training_data[self.features].to_numpy()[None, :, :])
            self.targets_reconstruction.adapt(training_data[self.targets])
        else:
            self.features_normalization.adapt(training_data.map(lambda x, _: x).batch(32))
            self.targets_reconstruction.adapt(training_data.map(lambda _, y: y))
        self.compile(**self.compilation_kwargs)

    def trim_to_forecast(self, data):
        """Trims the data to the appropriate size for the model.
        """
        rem = ((len(data) - self.dataset_maker.input_size) % self.prediction_shift)
        if rem != 0:
            return data.iloc[:-rem]
        return data

    def train(self, data, epochs, **kwargs):
        dataset = self.dataset_maker.supervised_dataset(data).shuffle(10_000)
        validation_dataset = dataset.take(512)
        training_dataset = dataset.skip(512)
        self.adapt_training(training_dataset)
        return self.fit(training_dataset.shuffle(10_000).batch(32), validation_data=validation_dataset.batch(32), epochs=epochs, **kwargs)

    def forecast(self, data):
        """Returns the predictions of the model on a dataset.
        """
        data = self.trim_to_forecast(data)
        raw_results = self.predict(
            self.dataset_maker.input_dataset(data[self.features]).batch(32)).reshape((-1, len(self.targets)))
        dates = self.dataset_maker.continuous_prediction_dates(data)
        return pd.DataFrame(raw_results, index=dates, columns=self.targets).rename_axis(index="date_time")


class DummyModel(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_days = 3
        self.output_days = 1
        self.output_hours = int(self.output_days * 24)
        self.offset_hours = (self.input_days - self.output_days) * 24
        self.shift_hours = self.output_hours
        self.input_spec = keras.layers.InputSpec(shape=(None, int(self.input_days * 24), 8))
        self.flatten = keras.layers.Reshape(target_shape=(-1,), input_shape=(int(self.input_days * 24), 8))
        self.fc = keras.layers.Dense(self.output_hours * len(targets))
        self.reshape = keras.layers.Reshape((self.output_hours, len(targets)))

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.fc(x)
        return self.reshape(x)
