import numpy as np
from .base_policy import BasePolicy
from .policy_factory import register_policy

@register_policy('thompson', 'thompson-sampling')
class ThompsonSamplingPolicy(BasePolicy):
    def __init__(self, num_actions, **config):
        super().__init__(num_actions, config)
        self.alpha = np.ones(num_actions)  # Parámetro para éxitos
        self.beta = np.ones(num_actions)   # Parámetro para fracasos

    def select_action(self, predict_fn=None, **predict_args):
        sampled_values = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_values)

    def update(self, episode, action, reward):
        if self.update_frequency == 'episode':
            if self.current_episode != episode:
                self.current_episode = episode
            else:
                return
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
