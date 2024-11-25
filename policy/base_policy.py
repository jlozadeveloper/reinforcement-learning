from abc import ABC, abstractmethod

class BasePolicy(ABC):
    def __init__(self, num_actions, config:dict):
        self.num_actions = num_actions
        self.update_frequency = config.get('update-frequency','step')
        self.config = config
        self.current_episode = 0

    @abstractmethod
    def select_action(self, predict_fn=None, **predict_args):
        pass

    def update(self, episode, action, reward):
        pass  # para políticas que necesitan actualizarse (como epsilon-greedy)

    def get_metrics(self):
        pass