import keras


class PollutionRnn(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_days = 7
        self.output_days = 1 / 24
        self.output_hours = 1
        self.offset_hours = int(24 * self.input_days) - 1
        self.shift_hours = 1
        self.input_spec = keras.layers.InputSpec(shape=(None, int(self.input_days * 24), 8))
        self.lstm = keras.layers.LSTM(512, return_sequences=False)
        self.fc = keras.layers.Dense(16, activation="relu")
        self.out = keras.layers.Dense(self.output_hours * 3)
        self.reshape = keras.layers.Reshape((self.output_hours, 3))

    def call(self, inputs, *args, **kwargs):
        x = self.lstm(inputs, *args, **kwargs)
        x = self.fc(x, *args, **kwargs)
        x = self.out(x, *args, **kwargs)
        return self.reshape(x)
