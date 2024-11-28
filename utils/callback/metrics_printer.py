import numpy as np
from typing import Tuple
from keras.src import callbacks

class MetricsPrinter(callbacks.Callback):
    
    def __init__(self, *args: Tuple[str, str]):
        """
        Clase encargada de imprimir en consola las métricas recogidas

        Args:
            *args: Una secuencia de tuplas, donde cada tupla contiene:
                - nombre_final (str): El nombre final de la métrica en pantalla.
                - nombre_metrica (str): La métrica que se desea imprimir.
        Ejemplo:
            mi_objeto = MetricsPrinter(
                ("Reward", "reward"),
                ("Pérdida", "loss")
            )
        """
        super().__init__()
        self.metrics = []
        for nombre_final, nombre_metrica in args:
            self.metrics.append({
                'name': nombre_final,
                'metric': nombre_metrica
            })
        
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
        pass
    
    def on_train_batch_begin(self, batch, logs=None): # 3
        pass

    def on_train_batch_end(self, batch, logs=None): # 4
        pass

    def on_epoch_end(self, epoch, logs=None): # 5
        # for k, output in enumerate(self.output_name):
        #     print(', ', end='') if k>0 else None
        #     print(f'{output}: {logs[self.input_name[k]]}', end='')
        for key, metric in enumerate(self.metrics):
            print(', ', end='') if key>0 else None
            print(f'{metric['name']}: {logs[metric['metric']]}', end='')
        print('') # just to print last '\n' (with auto end)

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

    
    
    

    