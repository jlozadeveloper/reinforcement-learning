from keras.src import callbacks

class MetricsUpdater(callbacks.Callback):
    
    def __init__(self):
        super().__init__()
        self._custom_metrics = []
        
    def set_model(self, model):
        super().set_model(model)
        self.find_custom_metrics()
        
    def find_custom_metrics(self):
        metrics = self.model._compile_metrics._user_metrics
        self._custom_metrics = [metric for metric in metrics if hasattr(metric, 'custom_update_state')]
            

        
    
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
        pass
    
    def on_train_batch_begin(self, batch, logs=None): # 3
        for metric in self._custom_metrics:
            metric.custom_update_state(**logs)

    def on_train_batch_end(self, batch, logs=None): # 4
        pass

    def on_epoch_end(self, epoch, logs=None): # 5
        for metric in self._custom_metrics:
            metric.custom_reset_state()


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

    
    
    

    