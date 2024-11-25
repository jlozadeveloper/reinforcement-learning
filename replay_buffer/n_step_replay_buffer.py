import numpy as np
from .base_replay_buffer import BaseReplayBuffer
from .replay_buffer_factory import register_buffer

@register_buffer('nstep', 'n-step', 'nstep-buffer')
class NStepReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity, n_steps=3, gamma=0.99):
        super().__init__(capacity)
        self.n_steps = n_steps
        self.gamma = gamma
        self._buffer = []
        self._position = 0
        self._current = None

    def add(self, state, action, reward, next_state, terminated, truncated):
        # Create new _current element or add reward and next_state to existing one
        if self._current is None:
            self._current = {
                "state": state,
                "action": action,
                "rewards": [reward],
                "next_states": [next_state],
                "terminated": terminated,
                "truncated": truncated
            }
        else:
            self._current['rewards'].append(reward)
            self._current['next_states'].append(next_state)

        # check if _current element is complete (has n-steps elements)
        if len(self._current['rewards']) >= self.n_steps:
            # recalculate rewards and next_state, and add final element to main buffer
            total_reward = 0
            gamma = 1
            for r in self._current["rewards"]:
                total_reward += gamma * r
                gamma *= self.gamma
            self._current["reward"] = total_reward
            self._current["next_state"] = self._current["next_states"][-1]
            del self._current["rewards"]
            del self._current["next_states"]

            # set final element and reset _current
            element = self._current
            self._current = None
        else:
            return

        if len(self._buffer) < self._capacity:
            self._buffer.append(element)
        else:
            self._buffer[self._position] = element
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        return batch, indices, None

    def __len__(self):
        return len(self._buffer)