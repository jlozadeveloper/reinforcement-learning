import numpy as np
from typing import Tuple, Callable
from keras.src import callbacks

class GenericMetrics(callbacks.Callback):
    
    def __init__(self, *args: Tuple[str, str, Callable]):
        """
        Esta clase representa una colección de métricas.

        Args:
            *args: Una secuencia de tuplas, donde cada tupla contiene:
                - nombre_final (str): El nombre final de la métrica por episodio.
                - nombre_metrica (str): El nombre de la métrica por batch.
                - funcion: Función a aplicar (del paquete numpy) a los valores de los batches para calcular la métrica del episodio.

        Ejemplo:
            mi_objeto = GenericMetrics(
                ("metrica1", "final1", 'mean'),
                ("metrica2", "final2", 'sum')
            )
        """
        super().__init__()
        self.metrics = []
        self.batches_metrics = {}
        for nombre_final, nombre_metrica, fn_name in args:
            fn = getattr(np, fn_name, None)
            if fn:
                self.metrics.append({
                    'name': nombre_final,
                    'metric': nombre_metrica,
                    'function': fn
                })
            else:
                raise Warning(f"Function '{fn_name}' from numpy not found.")
        
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
        self.batches_metrics = {metric['name']:[] for metric in self.metrics}
    
    def on_train_batch_begin(self, batch, logs=None): # 3
        pass

    def on_train_batch_end(self, batch, logs=None): # 4
        # for k, input in enumerate(self.input_name):
        #     self.batches_metrics[self.output_name[k]].append(logs[input])
        for metric in self.metrics:
            self.batches_metrics[metric['name']].append(logs[metric['metric']])

    def on_epoch_end(self, epoch, logs=None): # 5
        # for k, output in enumerate(self.batches_metrics):
        #     logs[output] = self.fn[k](self.batches_metrics[output])
        for metric in self.metrics:
            logs[metric['name']] = metric['function'](self.batches_metrics[metric['name']])

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

    
    
    

    