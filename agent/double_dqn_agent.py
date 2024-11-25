import random
import numpy as np
from keras import Model, models
from gymnasium import Env
from .agent_factory import register_agent
from .base_agent import BaseAgent
from .dqn_agent import DQNAgent
from replay_buffer import BaseReplayBuffer
from policy import BasePolicy
from utils import Config
from keras.api.callbacks import ModelCheckpoint, History, EarlyStopping


@register_agent('double-dqn')
class DoubleDQNAgent(DQNAgent):
  def __init__(self, env: Env, model: Model, policy: BasePolicy, replay_buffer: BaseReplayBuffer, **config):
    super().__init__(env, model, policy, replay_buffer, **config)

    self.target_update_freq = config.get('target-update-frequency', 10)
    self.tau =config.get('tau', 0.1)  # Update weight coefficient (adjust as needed)
    
    # Create a separate target model for Double DQN
    self.target_model: Model = models.clone_model(self.model)  # Clone the main model for the target
    self.target_model.set_weights(self.model.get_weights())  # Initialize weights the same


  def replay(self, episode, step):
    if len(self.replay_buffer) < self.train_start:
      return None
    
    if episode % self.target_update_freq == 0:
      self.update_target_network()
    
    # Sample minibatch from the memory
    minibatch, indices, weights = self.replay_buffer.sample(min(len(self.replay_buffer), self.batch_size))

    if minibatch is None:
      return None

    # Convert minibatch data to numpy arrays
    states = np.array([exp["state"] for exp in minibatch])
    actions = np.array([exp["action"] for exp in minibatch])
    rewards = np.array([exp["reward"] for exp in minibatch])
    next_states = np.array([exp["next_state"] for exp in minibatch])
    terminateds = np.array([exp["terminated"] for exp in minibatch])
    truncateds = np.array([exp["truncated"] for exp in minibatch])
    dones = np.logical_or(terminateds, truncateds) # New 'done' flag combining 'terminated' and 'truncated'

    # Obtener las predicciones de Q-values para todos los estados en el batch
    qvalues = self.model.predict_on_batch(states)

    # Predict Q-values for next states using the main model
    next_qvalues = self.model.predict_on_batch(next_states)

    # Select the best action for each next state using the main model (for Double DQN)
    best_next_actions = np.argmax(next_qvalues, axis=1)

    # Predict Q-values for next states and selected actions using the target model
    target_qvalues = self.target_model.predict_on_batch(next_states)
    target_qvalues = target_qvalues[np.arange(len(minibatch)), best_next_actions]

    # Calculate target Q-values with Double DQN approach
    targets = rewards + self.gamma * target_qvalues * (1 - dones)

    # calcular la diferencia entre valores objetivo y valores actuales
    td_errors = targets - qvalues[np.arange(len(minibatch)), actions]

    # Update Q-values for the main model
    qvalues[np.arange(len(minibatch)), actions] = targets

    self.replay_buffer.update(indices, td_errors)

    # Train the main model
    metrics = self.model.fit(states, qvalues, verbose=0, sample_weight=weights, batch_size=len(qvalues), shuffle=False)

    return metrics

  def update_target_network(self):
    # This method can define a soft update or a hard update strategy
    # A common approach is a soft update with a coefficient (e.g., tau)
    main_weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()
    new_weights = [self.tau * w1 + (1 - self.tau) * w2 for w1, w2 in zip(main_weights, target_weights)]
    self.target_model.set_weights(new_weights)