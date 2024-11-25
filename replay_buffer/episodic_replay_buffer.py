import numpy as np
from .base_replay_buffer import BaseReplayBuffer
from .replay_buffer_factory import register_buffer



###
# In this buffer, batch_size and capacity means number of complete episodes, not number of steps.
#
@register_buffer('episodic', 'episodic-buffer')
class EpisodicReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self._episodes = []
        self._position = 0
        self._current_episode = []
        

    def add(self, state, action, reward, next_state, terminated, truncated):
        self._current_episode.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "terminated": terminated,
            "truncated": truncated
        })

        if terminated or truncated:
            if len(self._episodes) < self._capacity:
                self._episodes.append(self._current_episode)
            else:
                self._episodes[self._position] = self._current_episode
            self._position = (self._position + 1) % self._capacity
            self._current_episode = []

    def sample(self, batch_size):
        if len(self._current_episode) != 0:
            return None
        indices = np.random.choice(len(self._episodes), size=batch_size, replace=False)
        
        # sampled_episodes = np.array(self._episodes)[np.random.choice(len(self._episodes), size=min(len(self._episodes), batch_size), replace=False), :]
        # batch = [transition for episode in sampled_episodes for transition in episode]
        batch = [self._episodes[i] for i in indices for _ in self._episodes[i]]
        return batch, indices, None

    def __len__(self):
        # return sum(len(episode) for episode in self._episodes)
        return len(self._episodes)