import numpy as np
from keras.src import callbacks

class GenericMetrics(callbacks.Callback):
    
    def __init__(self, output_name=[], input_name=[], fn=[]):
        super().__init__()
        self.output_name = output_name
        self.input_name = input_name
        self.fn = fn
        self.batches_metrics = {}
        
    def set_model(self, model):
        super().set_model(model)
    
    # sobreescribir las funciones de callback
    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass
        # guardar el modelo en un path dado
        # self.model.save(self.model_path)
    
    def on_train_begin(self, logs=None): # 1
        pass

    def on_epoch_begin(self, epoch, logs=None): # 2
        self.batches_metrics = {name:[] for name in self.output_name}
    
    def on_train_batch_begin(self, batch, logs=None): # 3
        pass

    def on_train_batch_end(self, batch, logs=None): # 4
        for k, input in enumerate(self.input_name):
            self.batches_metrics[self.output_name[k]].append(logs[input])

    def on_epoch_end(self, epoch, logs=None): # 5
        for k, output in enumerate(self.batches_metrics):
            logs[output] = self.fn[k](self.batches_metrics[output])

    def on_train_end(self, logs=None): # 6
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

    
    
    

    