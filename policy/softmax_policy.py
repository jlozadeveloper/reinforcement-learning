import tensorflow as tf, numpy as np
from .base_policy import BasePolicy
from .policy_factory import register_policy

@register_policy('softmax', 'boltzmann')
class SoftmaxPolicy(BasePolicy):
    def __init__(self, num_actions, **config):
        super().__init__(num_actions, config)
        self.temperature = config.get('temperature', 1.0)  # Parámetro de temperatura
        self.temperature_min = config.get('temperature-min', 0.001)
        self.temperature_decay = config.get('temperature-decay', 9e6)

    def select_action(self, predict_fn=None, **predict_args):
        q_values = predict_fn(**predict_args)[0]
        # # Aplicar softmax con temperatura
        # exp_q = np.exp(q_values / self.temperature)
        # probabilities = exp_q / np.sum(exp_q)
        probabilities = tf.nn.softmax(q_values / self.temperature).numpy()
        return np.random.choice(self.num_actions, p=probabilities)

    def update(self, episode, action, reward):
        if self.update_frequency == 'episode':
            if self.current_episode != episode:
                self.current_episode = episode
            else:
                return
        if self.temperature > self.temperature_min:
            self.temperature = max(self.temperature * self.temperature_decay, self.temperature_min)

    def get_metrics(self):
        return {'temperature': self.temperature}
