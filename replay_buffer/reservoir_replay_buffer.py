import numpy as np
from .base_replay_buffer import BaseReplayBuffer
from .replay_buffer_factory import register_buffer


@register_buffer('reservoir', 'reservoir-buffer')
class ReservoirReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self._buffer = []
        self._count = 0

    def add(self, state, action, reward, next_state, terminated, truncated):
        element = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "terminated": terminated,
            "truncated": truncated
        }
        if len(self._buffer) < self._capacity:
            self._buffer.append(element)
        else:
            idx = np.random.randint(0, self._count + 1)
            if idx < self._capacity:
                self._buffer[idx] = element
        self._count += 1

    def sample(self, batch_size):
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        return batch, indices, None

    def __len__(self):
        return len(self._buffer)