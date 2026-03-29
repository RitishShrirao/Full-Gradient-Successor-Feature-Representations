import numpy as np
from collections import defaultdict
import torch

class ReplayBuffer:
    def __init__(self, n_samples=1000000, n_batch=32):
        self.n_samples = n_samples
        self.n_batch = n_batch
        self.buffer = []
        self.index = 0
        self.size = 0
    
    def reset(self):
        self.buffer = []
        self.index = 0
        self.size = 0
    
    def append(self, state, action, reward, next_state, gamma):
        data = (state, action, reward, next_state, gamma)
        if self.size < self.n_samples:
            self.buffer.append(data)
            self.size += 1
        else:
            self.buffer[self.index] = data
        self.index = (self.index + 1) % self.n_samples

    def replay(self):
        if self.size < self.n_batch: return None
        indices = np.random.randint(0, self.size, size=self.n_batch)
        batch = [self.buffer[i] for i in indices]
        return self._unpack(batch)

    def sample_pivot(self):
        """Samples a single transition to serve as a pivot (s, a)."""
        if self.size == 0: return None
        idx = np.random.randint(0, self.size)
        return self.buffer[idx] # Returns (s, a, r, s', g)

    def _unpack(self, batch):
        states, actions, rewards, next_states, gammas = zip(*batch)
        return (np.vstack(states), np.array(actions), np.vstack(rewards), 
                np.vstack(next_states), np.array(gammas))

class ConditionalReplayBuffer(ReplayBuffer):
    """
    Maps (state_bytes, action) -> list of indices in the buffer.
    """
    def __init__(self, *args, **kwargs):
        super(ConditionalReplayBuffer, self).__init__(*args, **kwargs)
        self.transition_map = defaultdict(list)

    def reset(self):
        super(ConditionalReplayBuffer, self).reset()
        self.transition_map.clear()

    def _get_key(self, state, action):
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        return (state.tobytes(), int(action))

    def append(self, state, action, reward, next_state, gamma):
        data = (state, action, reward, next_state, gamma)
        idx = self.index
        if self.size == self.n_samples:
            old_state, old_action, _, _, _ = self.buffer[idx]
            old_key = self._get_key(old_state, old_action)
            if idx in self.transition_map[old_key]:
                self.transition_map[old_key].remove(idx)
            if not self.transition_map[old_key]:
                del self.transition_map[old_key]

        if self.size < self.n_samples:
            self.buffer.append(data)
            self.size += 1
        else:
            self.buffer[idx] = data

        key = self._get_key(state, action)
        self.transition_map[key].append(idx)
        self.index = (self.index + 1) % self.n_samples

    def sample_conditional(self, state, action, n_averaging):
        """
        Retrieves n_averaging transitions for the specific (state, action).
        """
        key = self._get_key(state, action)
        indices = self.transition_map.get(key, [])
        
        if len(indices) == 0: 
            return None
            
        if len(indices) >= n_averaging:
            selected_idx = np.random.choice(indices, n_averaging, replace=False)
        else:
            # Sample with replacement if we not enough samples
            selected_idx = np.random.choice(indices, n_averaging, replace=True)
            
        batch = [self.buffer[i] for i in selected_idx]
        return self._unpack(batch)