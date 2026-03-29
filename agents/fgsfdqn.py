import numpy as np
import torch
from agents.sfdqn import SFDQN

class FGSFDQN(SFDQN):
    def __init__(self, deep_sf, buffer, *args, 
                 algorithm='alg1',
                 n_averaging=1,
                 **kwargs):
        super(FGSFDQN, self).__init__(deep_sf, buffer, *args, **kwargs)
        self.algorithm = algorithm
        self.n_averaging = n_averaging

    def _get_gpi_policy(self, state_enc, task_reward_idx):
        if state_enc.ndim == 1:
            state_enc = state_enc.reshape(1, -1)
        # Returns (q_values, policy_indices)
        _, c = self.sf.GPI(state_enc, task_reward_idx, update_counters=False)
        return c.flatten()

    def _update_batch_grouped_by_prior(self, batch, task_i):
        states, actions, rewards, next_states, gammas = batch
        batch_size = states.shape[0]

        # Determine optimal c for every next_state in the batch using current task i
        c_indices = self._get_gpi_policy(next_states, task_i)

        # Group indices by their optimal policy c, map: policy_idx -> list of batch_indices
        groups = {}
        for idx, c in enumerate(c_indices):
            if c not in groups: groups[c] = []
            groups[c].append(idx)

        # Perform Update for each group
        for c, indices in groups.items():
            sub_batch = (
                states[indices],
                actions[indices],
                rewards[indices],
                next_states[indices],
                gammas[indices]
            )
            
            # Identify the prior task to update
            # If c == task_i, we only update task_i
            task_c = c if c != task_i else None
            self.sf.update_single_sample(sub_batch, task_i, task_c)

    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        # Store transition + reward update for all algorithms
        phi = self.phi(s, a, s1)
        if isinstance(phi, torch.Tensor):
            phi = phi.detach().cpu().numpy()
        phi_target = self.sf.encode_transition_feature(phi)
        self.sf.update_reward(phi, r, self.task_index)
        self.buffer.append(s_enc, a, phi_target, s1_enc, gamma)

        if self.algorithm == 'alg1':
            # Algorithm 1: Sequential, single-sample update
            batch = self.buffer.replay()
            if batch:
                self._update_batch_grouped_by_prior(batch, self.task_index)
            return

        if self.algorithm == 'alg4':
            # Algorithm 4: Algorithm 1 + averaging (as in Algorithm 3)
            # Uses a pivot (s,a) and a conditional batch to compute an averaged target
            # and an averaged prior-policy index.
            pivot = self.buffer.sample_pivot()
            if not pivot:
                return

            p_s, p_a, _, _, _ = pivot
            cond_batch = self.buffer.sample_conditional(p_s, p_a, self.n_averaging)
            if not cond_batch:
                return

            _, _, _, c_next_states, _ = cond_batch
            c = self.sf.get_averaged_gpi_policy_index(c_next_states, self.task_index)
            task_c = c if c != self.task_index else None
            self.sf.update_averaged(p_s, p_a, cond_batch, self.task_index, task_c)
            return

        # For Alg 2/3 (randomized training), updates happen inside train_randomized
        return

    def train_randomized(self, train_tasks, n_total_steps, viewers=None, n_view_ev=None):
        if viewers is None: viewers = [None] * len(train_tasks)
        self.reset()
        for t in train_tasks: self.add_training_task(t)
            
        for i in range(len(train_tasks)):
            self.set_active_training_task(i, True)
            self.active_task.initialize()

        
        self.set_active_training_task(0)
        for _ in range(self.buffer.n_batch): self.next_sample()
        
        for t in range(n_total_steps):
            if (t%100 == 0):
                print(t)
            i = np.random.randint(len(train_tasks))
            self.set_active_training_task(i, False)
            self.next_sample(viewers[i], n_view_ev)
            
            if self.algorithm == 'alg2':
                # Algorithm 2: Randomized, Single Sample
                batch = self.buffer.replay()
                if batch:
                    # Group by optimal Prior c
                    self._update_batch_grouped_by_prior(batch, i)
                    
            elif self.algorithm == 'alg3':
                # Algorithm 3: Randomized, Averaged Target
                # This operates on a fixed (s,a) pivot, so c is consistent.
                pivot = self.buffer.sample_pivot()
                if pivot:
                    p_s, p_a, _, _, _ = pivot
                    
                    # Get N samples for this specific (s,a)
                    cond_batch = self.buffer.sample_conditional(p_s, p_a, self.n_averaging)
                    
                    if cond_batch:
                        _, _, _, c_next_states, _ = cond_batch
                        
                        c = self.sf.get_averaged_gpi_policy_index(c_next_states, i)
                        
                        task_c = c if c != i else None
                        self.sf.update_averaged(p_s, p_a, cond_batch, i, task_c)