import keras


class PollutionNet(keras.Model):
    def __init__(self, n_features, n_targets, input_days, output_days, offset_hours, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features = n_features
        self.n_targets = n_targets
        self.input_days = input_days
        self.input_hours = round(self.input_days * 24)
        self.output_days = output_days
        self.output_hours = round(self.output_days * 24)
        self.offset_hours = offset_hours
        self.input_spec = keras.layers.InputSpec(
            shape=(None, self.input_hours, n_features))


class DirectRnn(PollutionNet, keras.Sequential):
    """An RNN-based model, where all the rnn layers return sequences as long as the input
    """
    def __init__(self, n_features, n_targets, rnn_class, n_recursive_units, input_days, output_days, offset_hours=None, n_intermediate_fc_units=[], rnn_kwargs={}, *args, **kwargs):
        if offset_hours is None:
            offset_hours = round(input_days * 24)
        super().__init__(n_features, n_targets, input_days, output_days, offset_hours, *args, **kwargs)
        if not hasattr(n_recursive_units, "__iter__"):
            n_recursive_units = [n_recursive_units]
        for i, n_units in enumerate(n_recursive_units[:-1]):
            self.add(rnn_class(n_units, return_sequences=True, **rnn_kwargs, name=f"{self.name}_{rnn_class.__name__}_{i}"))
        self.add(rnn_class(n_recursive_units[-1], return_sequences=True, name=f"{self.name}_{rnn_class.__name__}_last"))
        self.cropping = keras.layers.Cropping1D((self.input_hours - self.output_hours, 0), name=f"{self.name}_cropping")
        self.add(self.cropping)
        for n_units in n_intermediate_fc_units:
            self.add(keras.layers.Dense(n_units, name=f"{self.name}_intermediate_dense_{n_units}"))
        self.out = keras.layers.Dense(n_targets, name=f"{self.name}_out")
        self.add(self.out)
