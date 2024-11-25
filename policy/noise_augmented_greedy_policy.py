import numpy as np
from .base_policy import BasePolicy
from .policy_factory import register_policy

@register_policy('noise_greedy', 'noise-augmented-greedy')
class NoiseAugmentedGreedyPolicy(BasePolicy):
    def __init__(self, num_actions, **config):
        super().__init__(num_actions, config)
        self.noise_scale = config.get('noise_scale', 0.1)  # Escala del ruido gaussiano

    def select_action(self, predict_fn=None, **predict_args):
        q_values = predict_fn(**predict_args)
        noisy_q_values = q_values + np.random.normal(0, self.noise_scale, size=q_values.shape)
        return np.argmax(noisy_q_values)

    def update(self, episode, action, reward):
        pass  # Esta política no requiere actualización interna
