from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from keras import Model, models
from gymnasium import Env
from policy import BasePolicy
from replay_buffer import BaseReplayBuffer
from utils import Config

class BaseAgent(ABC):
    def __init__(self, env: Env, model: Model, policy:BasePolicy, replay_buffer:BaseReplayBuffer, **config):
        self.env = env
        self.model = model
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.config = config

        self.input_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        self.train_start = config.get('train-start', 1)
        self.batch_size = config.get('batch-size', 64)
        

    @abstractmethod
    def replay(self, episode, step):
        pass

    @abstractmethod
    def episode_ended(self, episode, episode_metrics):
        pass

    def select_action(self, state:npt.NDArray):
        state = state.reshape(1, *self.input_shape)
        return self.policy.select_action(self.model.predict, x=state, verbose=0)

    def remember(self, state, action, reward, next_state, terminated, truncated):
        self.replay_buffer.add(state, action, reward, next_state, terminated, truncated)

    def update_policy(self, episode, action, reward):
        policy_metrics = self.policy.get_metrics()
        if len(self.replay_buffer) >= self.train_start:
            self.policy.update(episode, action, reward)
        return policy_metrics

    def load(self, name):
        self.model = models.load_model(name)

    def save(self, name):
        # self.save(f"{self.agent.env.unwrapped.spec.id}-{self.agent.__class__.__name__}.keras")
        self.model.save(name)