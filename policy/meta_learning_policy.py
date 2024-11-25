import numpy as np
from .base_policy import BasePolicy
from .policy_factory import register_policy

@register_policy('meta', 'meta-learning')
class MetaLearningPolicy(BasePolicy):
    def __init__(self, num_actions, **config):
        super().__init__(num_actions, config)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.q_values = np.zeros(num_actions)

    def select_action(self, predict_fn=None, **predict_args):
        exploration_prob = 1 / (1 + np.exp(-self.learning_rate * self.q_values))
        return np.random.choice(self.num_actions, p=exploration_prob / np.sum(exploration_prob))

    def update(self, episode, action, reward):
        if self.update_frequency == 'episode':
            if self.current_episode != episode:
                self.current_episode = episode
            else:
                return
        self.q_values[action] += self.learning_rate * (reward - self.q_values[action])
