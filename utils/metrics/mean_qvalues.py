from keras import Metric
from keras.src.metrics import Mean
from keras.src import ops

class MeanQValues(Mean):
    def __init__(self, name='test_metric', predicted=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.predicted = predicted

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.predicted:
            values = ops.cast(y_pred, "float32")
        else:
            values = ops.cast(y_true, "float32")
        return super().update_state(values, sample_weight=sample_weight)