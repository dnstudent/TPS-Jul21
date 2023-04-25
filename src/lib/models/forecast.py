import keras
from . import PollutionNet, DirectRnn







class DirectLSTM(DirectRnn):
    def __init__(self, n_features, n_targets, n_recursive_units, input_days, output_days, offset_hours=None, n_intermediate_fc_units=None, *args, **kwargs):
        if n_intermediate_fc_units is None:
            n_intermediate_fc_units = []
        super().__init__(n_features, n_targets, keras.layers.LSTM, n_recursive_units, input_days, output_days, offset_hours, n_intermediate_fc_units, *args, **kwargs)


class DirectGRU(DirectRnn):
    def __init__(self, n_features, n_targets, n_recursive_units, input_days, output_days, offset_hours=None, n_intermediate_fc_units=None, *args, **kwargs):
        if n_intermediate_fc_units is None:
            n_intermediate_fc_units = []
        super().__init__(n_features, n_targets, keras.layers.GRU, n_recursive_units, input_days, output_days, offset_hours, n_intermediate_fc_units, *args, **kwargs)


class MultiHeadRnn(PollutionNet):
    def __init__(self, n_features, n_targets, layer_class, n_backbone_units, n_head_units, input_days, output_days, offset_hours):
        super().__init__(n_features, input_days, output_days, offset_hours)
        if offset_hours is None:
            self.offset_hours = self.input_hours - 1
        self.backbone = layer_class(n_backbone_units, return_sequences=True)
        for head in range(n_targets):
            pass
