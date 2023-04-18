import keras
from . import PollutionNet, DirectRnn


# class SimpleRnn(PollutionNet, keras.Sequential):
#     def __init__(self, n_features, n_targets, layer_class, n_recursive_units, input_days, output_days, offset_hours):
#         super().__init__(n_features, n_targets, input_days, output_days, offset_hours)
#         if offset_hours is None:
#             self.offset_hours = self.input_hours - self.output_hours
#         self.lstm = layer_class(n_recursive_units, return_sequences=True)
#         self.add(self.lstm)
#         self.cropping = keras.layers.Cropping1D((self.offset_hours, 0))
#         self.add(self.cropping)
#         self.out = keras.layers.Dense(n_targets)
#         self.add(self.out)


# class SimplePollutionLSTM(SimpleRnn):
#     def __init__(self, n_features, n_targets, n_recursive_units, input_days, output_days, offset_hours):
#         super().__init__(n_features, n_targets, keras.layers.LSTM, n_recursive_units, input_days, output_days, offset_hours)


# class SimplePollutionGRU(SimpleRnn):
#     def __init__(self, n_features, n_targets, n_recursive_units, input_days, output_days, offset_hours):
#         super().__init__(n_features, n_targets, keras.layers.GRU, n_recursive_units, input_days, output_days, offset_hours)

class DirectLSTM(DirectRnn):
    def __init__(self, n_features, n_targets, n_recursive_units, input_days, output_days, offset_hours=None, n_intermediate_fc_units=[], *args, **kwargs):
        super().__init__(n_features, n_targets, keras.layers.LSTM, n_recursive_units, input_days, output_days, offset_hours, n_intermediate_fc_units, *args, **kwargs)


class DirectGRU(DirectRnn):
    def __init__(self, n_features, n_targets, n_recursive_units, input_days, output_days, offset_hours=None, n_intermediate_fc_units=[], *args, **kwargs):
        super().__init__(n_features, n_targets, keras.layers.GRU, n_recursive_units, input_days, output_days, offset_hours, n_intermediate_fc_units, *args, **kwargs)


class MultiHeadRnn(PollutionNet):
    def __init__(self, n_features, n_targets, layer_class, n_backbone_units, n_head_units, input_days, output_days, offset_hours):
        super().__init__(n_features, input_days, output_days, offset_hours)
        if offset_hours is None:
            self.offset_hours = self.input_hours - 1
        self.backbone = layer_class(n_backbone_units, return_sequences=True)
        for head in range(n_targets):
            pass
