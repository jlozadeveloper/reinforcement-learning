from keras import Metric
from keras.src.metrics import Mean
from keras.src import ops

class TestMetric(Mean):
    def __init__(self, name='test_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean_diff = self.add_variable(
            shape=(),
            initializer='zeros',
            aggregation='mean',
            name='mean_diff'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")

        # values = ops.abs(y_true - y_pred)
        values = y_true
        values = ops.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            sample_weight = ops.broadcast_to(
                sample_weight, ops.shape(values)
            )
            values = ops.multiply(values, sample_weight)
        self.mean_diff.assign(ops.mean(values))

    def reset_state(self):
        j=''
        return super().reset_state()

    def result(self):
        return self.mean_diff