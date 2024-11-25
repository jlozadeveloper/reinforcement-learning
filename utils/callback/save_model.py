from keras.src import callbacks

class SaveModel(callbacks.Callback):
    
    def __init__(self, model_path=None):
        super().__init__()
        if model_path is None:
            model_path = 'model.h5'
        self.model_path = model_path
    
    def set_model(self, model):
        super().set_model(model)
        j=''
    
    # sobreescribir las funciones de callback
    def on_batch_begin(self, batch, logs=None):
        print(f'on_batch_begin: {batch}')
        print(logs)
        pass

    def on_batch_end(self, batch, logs=None):
        print(f'on_batch_end: {batch}')
        print(logs)
        pass
        # guardar el modelo en un path dado
        # self.model.save(self.model_path)
    
    def on_train_begin(self, logs=None): # 1
        print(f'on_train_begin')
        print(logs)
        pass

    def on_epoch_begin(self, epoch, logs=None): # 2
        print(f'on_epoch_begin: {epoch}')
        print(logs)
        pass
    
    def on_train_batch_begin(self, batch, logs=None): # 3
        print(f'on_train_batch_begin: {batch}')
        print(logs)
        pass

    def on_train_batch_end(self, batch, logs=None): # 4
        print(f'on_train_batch_end: {batch}')
        print(logs)
        pass

    def on_epoch_end(self, epoch, logs=None): # 5
        print(f'on_epoch_end: {epoch}')
        print(logs)
        pass

    def on_train_end(self, logs=None): # 6
        print(f'on_train_end')
        print(logs)
        pass

    def on_test_begin(self, logs=None):
        print(f'on_test_begin')
        print(logs)
        pass

    def on_test_end(self, logs=None):
        print(f'on_test_end')
        print(logs)
        pass

    def on_test_batch_begin(self, batch, logs=None):
        print(f'on_test_batch_begin: {batch}')
        print(logs)
        pass

    def on_test_batch_end(self, batch, logs=None):
        print(f'on_test_batch_end: {batch}')
        print(logs)
        pass

    def on_predict_begin(self, logs=None):
        print(f'on_predict_begin')
        print(logs)
        pass

    def on_predict_end(self, logs=None):
        print(f'on_predict_end')
        print(logs)
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        print(f'on_predict_batch_begin: {batch}')
        print(logs)
        pass

    def on_predict_batch_end(self, batch, logs=None):
        print(f'on_predict_batch_end: {batch}')
        print(logs)
        pass

    
    
    

    