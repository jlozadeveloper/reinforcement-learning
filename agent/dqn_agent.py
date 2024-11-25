import numpy as np
from keras import Model
from keras.src.callbacks import History
from gymnasium import Env
from .base_agent import BaseAgent
from .agent_factory import register_agent
from replay_buffer import BaseReplayBuffer
from policy import BasePolicy
from keras.src import callbacks as callbacks_module


@register_agent('dqn')
class DQNAgent(BaseAgent):
    def __init__(self, env:Env, model:Model, policy:BasePolicy, replay_buffer:BaseReplayBuffer, **config):
        super().__init__(env, model, policy, replay_buffer, **config)

        self.gamma = config.get('gamma', 0.95)  # discount rate

        


    def replay(self, episode, step):
        if len(self.replay_buffer) < self.train_start:
            return None
        # Randomly sample minibatch from the memory
        minibatch, indices, weights = self.replay_buffer.sample(min(len(self.replay_buffer), self.batch_size))

        if minibatch is None:
            return None

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        states = np.array([exp["state"] for exp in minibatch])
        actions = np.array([exp["action"] for exp in minibatch])
        rewards = np.array([exp["reward"] for exp in minibatch])
        next_states = np.array([exp["next_state"] for exp in minibatch])
        terminateds = np.array([exp["terminated"] for exp in minibatch])
        truncateds = np.array([exp["truncated"] for exp in minibatch])
        dones = np.logical_or(terminateds, truncateds) # New 'done' flag combining 'terminated' and 'truncated'

        # Obtener las predicciones de Q-values para todos los estados en el batch
        qvalues = self.model.predict_on_batch(states)
        next_qvalues = self.model.predict_on_batch(next_states)

        # Calcular los valores objetivo Q
        targets = rewards + self.gamma * np.amax(next_qvalues, axis=1) * (1 - dones) 

        # calcular la diferencia entre valores objetivo y valores actuales
        td_errors = targets - qvalues[np.arange(len(minibatch)), actions]
        
        # Actualizar los Q-values correspondientes a las acciones tomadas
        qvalues[np.arange(len(minibatch)), actions] = targets

        self.replay_buffer.update(indices, td_errors)

        # Entrenar el modelo
        # metrics:History = self.model.fit(states, qvalues, verbose=0, sample_weight=weights, batch_size=len(qvalues), shuffle=False, callbacks=self.callbacks)
        metrics:History = self.model.fit(states, qvalues, verbose=0, sample_weight=weights, batch_size=int(len(qvalues)/4), shuffle=False)
        
        return metrics
    
    def episode_ended(self, episode, episode_metrics):
        if episode_metrics is None:
            print("None metrics")