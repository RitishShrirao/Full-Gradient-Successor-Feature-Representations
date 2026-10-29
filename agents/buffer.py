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
    Supports continuous states by rounding to spatial bins (state_decimals).
    """
    def __init__(self, *args, state_decimals=2, **kwargs):
        super(ConditionalReplayBuffer, self).__init__(*args, **kwargs)
        self.transition_map = defaultdict(list)
        self.state_decimals = state_decimals
        self._reset_arrays()

    def _reset_arrays(self):
        # Array mirror of the buffer used by the vectorized sampler.
        self._key_ids = {}                                   # (state bytes, action) -> group id
        self._slot_key = np.full(self.n_samples, -1, dtype=np.int64)
        self._states = self._phis = self._nexts = None
        self._actions = np.zeros(self.n_samples, dtype=np.int64)
        self._gammas = np.zeros(self.n_samples, dtype=np.float32)
        self._groups = None                                  # cached (order, starts, counts)

    def reset(self):
        super(ConditionalReplayBuffer, self).reset()
        self.transition_map.clear()
        self._reset_arrays()

    def _get_key(self, state, action):
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state_arr = np.asarray(state).reshape(-1)
        if self.state_decimals is not None and np.issubdtype(state_arr.dtype, np.floating):
            # Round continuous states so nearby states cluster into the same key
            state_arr = np.round(state_arr, decimals=self.state_decimals)
        return (state_arr.tobytes(), int(action))

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

        s_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        phi_arr = np.asarray(reward, dtype=np.float32).reshape(-1)
        if self._states is None:
            self._states = np.zeros((self.n_samples, s_arr.size), dtype=np.float32)
            self._nexts = np.zeros((self.n_samples, s_arr.size), dtype=np.float32)
            self._phis = np.zeros((self.n_samples, phi_arr.size), dtype=np.float32)
        self._states[idx] = s_arr
        self._nexts[idx] = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self._phis[idx] = phi_arr
        self._actions[idx] = int(action)
        self._gammas[idx] = float(gamma)
        self._slot_key[idx] = self._key_ids.setdefault(key, len(self._key_ids))
        self._groups = None

        self.index = (self.index + 1) % self.n_samples

    def sample_pivot_with_min_samples(self, min_samples=1):
        """Samples a pivot (s, a) that has at least min_samples transitions."""
        if self.size == 0:
            return None
        for _ in range(20):
            idx = np.random.randint(0, self.size)
            item = self.buffer[idx]
            key = self._get_key(item[0], item[1])
            if len(self.transition_map.get(key, [])) >= min_samples:
                return item
        return self.sample_pivot()

    def sample_conditional(self, state, action, n_averaging, min_samples=1):
        """
        Retrieves n_averaging transitions for the specific (state, action).
        """
        key = self._get_key(state, action)
        indices = self.transition_map.get(key, [])
        
        if len(indices) < min_samples: 
            return None
            
        if len(indices) >= n_averaging:
            selected_idx = np.random.choice(indices, n_averaging, replace=False)
        else:
            selected_idx = np.random.choice(indices, n_averaging, replace=True)
            
        batch = [self.buffer[i] for i in selected_idx]
        return self._unpack(batch)

    def _group_index(self):
        """Slots sorted by (s, a) group: order, and per-group start offsets and counts."""
        if self._groups is None:
            keys = self._slot_key[:self.size]
            order = np.argsort(keys, kind='stable')
            counts = np.bincount(keys, minlength=len(self._key_ids))
            starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
            self._groups = (order, starts, counts)
        return self._groups

    def sample_batch_conditional(self, batch_size, n_averaging):
        """
        Samples a batch of batch_size pivots (s, a) uniformly from the buffer and, for each pivot,
        n_averaging transitions that start at the same (s, a): without replacement when the pivot
        has at least n_averaging transitions, with replacement otherwise. Vectorized over pivots.

        Returns:
            p_states: np.ndarray [B, state_dim]
            p_actions: np.ndarray [B]
            c_phis: np.ndarray [B, N, feat_dim]
            c_nexts: np.ndarray [B, N, state_dim]
            c_gammas: np.ndarray [B, N]
        """
        if self.size < self.n_batch or self.size == 0:
            return None

        B = min(batch_size, self.size)
        N = int(n_averaging)
        order, starts, counts = self._group_index()
        pivots = np.random.randint(0, self.size, size=B)
        group = self._slot_key[pivots]
        start, cnt = starts[group], counts[group]

        # Offsets inside each pivot's group. Floyd's algorithm gives a uniform subset without
        # replacement when cnt >= N; otherwise sample with replacement.
        offs = np.empty((B, N), dtype=np.int64)
        rand = np.random.random((B, N))
        full = cnt >= N
        offs[~full] = np.floor(rand[~full] * cnt[~full, None]).astype(np.int64)
        if full.any():
            c_full = cnt[full]
            chosen = np.empty((c_full.size, N), dtype=np.int64)
            for step in range(N):
                j = c_full - N + step
                t = np.floor(rand[full, step] * (j + 1)).astype(np.int64)
                if step > 0:
                    dup = (chosen[:, :step] == t[:, None]).any(axis=1)
                    t = np.where(dup, j, t)
                chosen[:, step] = t
            offs[full] = chosen
        members = order[start[:, None] + offs]  # [B, N] buffer slots

        return (
            self._states[pivots],
            self._actions[pivots],
            self._phis[members],
            self._nexts[members],
            self._gammas[members],
        )
