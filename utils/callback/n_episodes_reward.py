import numpy as np
from keras.src import callbacks

class NEpisodesReward(callbacks.Callback):
    
    def __init__(self, name=None, n_episodes=50, n_ep_fn=np.mean, batches_fn=np.sum):
        super().__init__()
        self.metric_name = name or f'{n_episodes}_ep_{n_ep_fn.__name__}'
        self.n_episodes = n_episodes
        self.n_ep_fn = n_ep_fn
        self.batches_fn = batches_fn
        self.batches_rewards = []
        self.episodes_rewards = []
        
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
        self.episodes_rewards = []
        pass

    def on_epoch_begin(self, epoch, logs=None): # 2
        self.batches_rewards = []
        pass
    
    def on_train_batch_begin(self, batch, logs=None): # 3
        pass

    def on_train_batch_end(self, batch, logs=None): # 4
        self.batches_rewards.append(logs['reward'])
        pass

    def on_epoch_end(self, epoch, logs=None): # 5
        self.episodes_rewards.append(self.batches_fn(self.batches_rewards))
        self.episodes_rewards = self.episodes_rewards[-1*self.n_episodes:]
        logs[self.metric_name] = self.n_ep_fn(self.episodes_rewards)
        pass

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

    
    
    

    