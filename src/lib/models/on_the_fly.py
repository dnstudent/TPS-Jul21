import keras
from . import DirectRnn


class OnTheFlyRnn(DirectRnn):
    def __init__(self, n_features, n_targets, rnn_class, n_recursive_units, input_days, n_intermediate_fc_units=None, rnn_kwargs=None, *args, **kwargs):
        if n_intermediate_fc_units is None:
            n_intermediate_fc_units = []
        if rnn_kwargs is None:
            rnn_kwargs = []
        super().__init__(n_features, n_targets, rnn_class, n_recursive_units, input_days, input_days, 0, n_intermediate_fc_units, rnn_kwargs, *args, **kwargs)


class OnTheFlyLSTM(DirectRnn):
    def __init__(self, n_features, n_targets, n_recursive_units, input_days, n_intermediate_fc_units=None, rnn_kwargs=None, *args, **kwargs):
        if n_intermediate_fc_units is None:
            n_intermediate_fc_units = []
        if rnn_kwargs is None:
            rnn_kwargs = {}
        super().__init__(n_features, n_targets, keras.layers.LSTM, n_recursive_units, input_days, input_days, 0, n_intermediate_fc_units, rnn_kwargs, *args, **kwargs)


class OnTheFlyGRU(DirectRnn):
    def __init__(self, n_features, n_targets, n_recursive_units, input_days, n_intermediate_fc_units=None, rnn_kwargs=None, *args, **kwargs):
        if n_intermediate_fc_units is None:
            n_intermediate_fc_units = []
        if rnn_kwargs is None:
            rnn_kwargs = {}
        super().__init__(n_features, n_targets, keras.layers.GRU, n_recursive_units, input_days, input_days, 0, n_intermediate_fc_units, rnn_kwargs, *args, **kwargs)