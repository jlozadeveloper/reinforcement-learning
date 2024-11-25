import numpy as np
from .base_policy import BasePolicy
from .policy_factory import register_policy

@register_policy('ucb', 'upper-confidence-bound')
class UCBPolicy(BasePolicy):
    def __init__(self, num_actions, **config):
        super().__init__(num_actions, config)
        self.c = config.get('c', 1.0)  # Parámetro de control para exploración
        self.action_counts = np.zeros(num_actions)  # Número de veces que se selecciona cada acción
        self.q_values = np.zeros(num_actions)  # Valores Q estimados para cada acción

    def select_action(self, predict_fn=None, **predict_args):
        total_counts = np.sum(self.action_counts) + 1  # Para evitar división por cero
        ucb_values = self.q_values + self.c * np.sqrt(np.log(total_counts) / (self.action_counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, episode, action, reward):
        self.action_counts[action] += 1
        if self.update_frequency == 'episode':
            if self.current_episode != episode:
                self.current_episode = episode
            else:
                return
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
