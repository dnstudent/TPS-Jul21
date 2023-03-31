from tensorflow import keras
from . import PollutionEstimator


class PollutionDensePredictor(PollutionEstimator):
    """A model that estimates pollution levels from weather data in a given time frame, meaning that for each row of data (= hourly measure) an istantaneous estimation is given.
    Uses a dense NN under the hood.
    """

    def __init__(self, features, targets, input_days, output_days, *args, **kwargs):
        super().__init__(features, targets, input_days,
                         output_days, offset_hours=0, *args, **kwargs)
        self.flatten = keras.layers.Reshape((-1,))
        self.dense = keras.layers.Dense(len(targets) * self.output_wsize)
        self.reshape = keras.layers.Reshape((self.output_wsize, len(targets)))

    def call(self, inputs):
        x = super().call(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        return self.reshape(x)


# class PollutionCNNPredictor(PollutionModel):
#     """A model that estimates pollution levels from weather data in a given time frame, meaning that for each row of data (= hourly measure) an istantaneous estimation is given.
#     Uses an RNN under the hood.
#     """
#     def __init__(self, features, targets, input_days, output_days, *args, **kwargs):
#         super().__init__(features, targets, input_days,
#                          output_days, offset_days=0, *args, **kwargs)
#         self.cnn = keras.layers.Conv1D(8, 4, padding="same", input_shape=(self.input_wsize, len(self.features)))
#         self.dense = keras.layers.Dense(self.output_wsize * len(targets))
#         self.reshape = keras.layers.Reshape((self.output_wsize, len(targets)))
#     def call(self, inputs):
#         x = super().call(inputs)
#         x = self.cnn(x)
#         x = self.dense(x)
#         return self.reshape(x)


class PollutionSimplePredictor(PollutionEstimator):
    """A model that estimates pollution levels from weather data in a given time frame, meaning that for each row of data (= hourly measure) an istantaneous estimation is given.
    Uses an RNN under the hood.
    """

    def __init__(self, features, targets, input_days, output_days, offset_days, *args, **kwargs):
        super().__init__(self, features, targets, input_days,
                         output_days, offset_days, *args, **kwargs)
        self.lstm = keras.layers.LSTM(len(targets), return_sequences=True)

    def call(self, inputs):
        return self.lstm(inputs)
