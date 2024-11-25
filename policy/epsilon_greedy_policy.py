import random, numpy as np
from utils import Config
from .base_policy import BasePolicy
from .policy_factory import register_policy

@register_policy('epsilon', 'epsilon-greedy')
class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, num_actions, **config:dict):
        super().__init__(num_actions, config)
        self.epsilon = config.get('epsilon', 1)
        self.epsilon_min = config.get('epsilon-min', 0.001)
        self.epsilon_decay = config.get('epsilon-decay', 0.99)

    def select_action(self, predict_fn=None, **predict_args):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        else:
            q_values = predict_fn(**predict_args)
            return np.argmax(q_values)

    def update(self, episode, action, reward):
        if self.update_frequency == 'episode':
            if self.current_episode != episode:
                self.current_episode = episode
            else:
                return
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def get_metrics(self):
        return {'epsilon': self.epsilon}