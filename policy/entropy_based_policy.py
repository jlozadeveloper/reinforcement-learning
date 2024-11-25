import numpy as np, tensorflow as tf
from .base_policy import BasePolicy
from .policy_factory import register_policy

@register_policy('entropy', 'entropy-based')
class EntropyBasedPolicy(BasePolicy):
    def __init__(self, num_actions, **config):
        super().__init__(num_actions, config)
        self.entropy_weight = config.get('entropy-weight', 1.0)  # Controla la importancia de la entropía

    def select_action(self, predict_fn=None, **predict_args):
        q_values = predict_fn(**predict_args)[0]
        # exp_q = np.exp(q_values * self.entropy_weight)
        # probabilities = exp_q / np.sum(exp_q)
        prob_dist = tf.nn.softmax(q_values * self.entropy_weight).numpy()

        return np.random.choice(self.num_actions, p=prob_dist)

    def update(self, episode, action, reward):
        pass  # En esta implementación, no hay actualizaciones dinámicas necesarias
