import torch
import torch.nn as nn
import numpy as np
from features.deep import DeepSF

class DeepFGSF(DeepSF):
    def __init__(self, learning_rate_prior=None, *args, **kwargs):
        super(DeepFGSF, self).__init__(*args, **kwargs)
        self.learning_rate_prior = learning_rate_prior if learning_rate_prior is not None else self.learning_rate

    def _set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
    def _to_tensor(self, x, dtype=torch.float):
        if not isinstance(x, torch.Tensor):
            return torch.from_numpy(x).to(dtype=dtype, device=self.device)
        return x.to(dtype=dtype, device=self.device)

    def _get_next_actions_gpi(self, next_states, policy_index):
        """
        Selects actions using GPI based on the current task's reward weights
        but evaluating over all policies.
        """
        # Get action that maximizes Q for the current task.
        q_gpi, _ = self.GPI(next_states, policy_index)   # [Batch, n_tasks, n_actions]
        if isinstance(q_gpi, torch.Tensor):
            return torch.argmax(torch.max(q_gpi, dim=1).values, dim=-1).long().to(self.device)

        next_actions = np.argmax(np.max(q_gpi, axis=1), axis=-1)
        return torch.from_numpy(next_actions).long().to(self.device)

    def _get_next_action_prior(self, next_states, policy_index):
        """
        Selects actions using the prior policy for a specific task k:
        a' = argmax_a (psi_k(s', a) * w_k)
        """
        model, _, _ = self.psi[policy_index]
        model.eval()
        
        # Get Features for policy k
        with torch.no_grad():
            psi = model(next_states) # [Batch, Actions, Features]
            
        # Score successor descriptors with the configured reward model.
        q_values = self.score_successor(
            psi,
            task_index=policy_index,
            w=self.fit_w[policy_index] if self.reward_model == 'linear' else None,
        )
        if not isinstance(q_values, torch.Tensor):
            q_values = torch.as_tensor(q_values, dtype=torch.float32, device=self.device)
        
        next_actions = torch.argmax(q_values, dim=1)
        return next_actions
    
    def get_averaged_gpi_policy_index(self, next_states, task_index):
        """
        Determines the prior policy 'c' by averaging the SFs of the batch 
        of next states and then applying GPI.
        """
        if not isinstance(next_states, torch.Tensor):
            next_states = self._to_tensor(next_states)

        best_policy_idx = -1
        best_q_val = -float('inf')

        # Evaluate every policy k
        for k in range(self.n_tasks):
            model, _, _ = self.psi[k]
            model.eval()
            with torch.no_grad():
                # Compute SFs for all samples in batch [N, Actions, Features]
                all_sfs = model(next_states)
                
                # Average the SFs across the batch dimension
                # This approximates E[xi(s', a')]
                avg_sf = torch.mean(all_sfs, dim=0) 
                
                # Compute Q-values for current task reward model.
                q_vals = self.score_successor(
                    avg_sf,
                    task_index=task_index,
                    w=self.fit_w[task_index] if self.reward_model == 'linear' else None,
                )
                if isinstance(q_vals, torch.Tensor):
                    max_q = float(torch.max(q_vals).item())
                else:
                    max_q = float(np.max(q_vals))
                
                if max_q > best_q_val:
                    best_q_val = max_q
                    best_policy_idx = k
                    
        return best_policy_idx

    def update_single_sample(self, transition, task_i, task_c):
        """
        Updates theta^i and theta^c using a single transition (s, a, s')
        """
        states, actions, phis, next_states, gammas = transition
        
        states = self._to_tensor(states)
        actions = self._to_tensor(actions, torch.long)
        phis = self._to_tensor(phis)
        next_states = self._to_tensor(next_states)
        gammas = self._to_tensor(gammas).view(-1, 1)
        
        indices = torch.arange(states.shape[0], device=self.device)
        
        # Identify tasks to update
        tasks_to_update = {task_i}
        if task_c is not None:
            tasks_to_update.add(task_c)
            
        total_loss = 0
        
        for k in tasks_to_update:
            model, _, optimizer = self.psi[k]
            
            # Update LR
            lr = self.learning_rate if k == task_i else self.learning_rate_prior
            self._set_lr(optimizer, lr)

            model.train()
            optimizer.zero_grad()

            if k==task_i:
                # Current task: Use GPI action
                next_acts = self._get_next_actions_gpi(next_states, task_i)
            else:
                # Prior Task
                next_acts = self._get_next_action_prior(next_states, k)
            
            # Prediction: xi_k(s, a)
            pred_all = model(states)
            xi_s = pred_all[indices, actions, :]
            
            # Target: phi + gamma * xi_k(s', a')
            # Full Gradient: Gradients flow through xi_k(s')
            pred_next_all = model(next_states)
            xi_next = pred_next_all[indices, next_acts, :]
            
            target = phis + gammas * xi_next
            
            diff = target - xi_s
            loss = 0.5 * (diff.pow(2)).mean()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def update_averaged(self, pivot_state, pivot_action, conditional_batch, task_i, task_c):
        """
        Implements averaging for bellman update over N samples.
        """
        _, _, c_phis, c_next_states, c_gammas = conditional_batch
        
        c_phis = self._to_tensor(c_phis)
        c_nexts = self._to_tensor(c_next_states)
        c_gammas = self._to_tensor(c_gammas).view(-1, 1)
        
        pivot_state_t = self._to_tensor(pivot_state)
        tasks_to_update = {task_i}
        if task_c is not None:
            tasks_to_update.add(task_c)

        for k in tasks_to_update:
            model, _, optimizer = self.psi[k]
            
            # Update LR
            lr = self.learning_rate if k == task_i else self.learning_rate_prior
            self._set_lr(optimizer, lr)

            model.train()
            optimizer.zero_grad()
            
            # xi_k(s, a) (Single value for the pivot)
            pred_pivot = model(pivot_state_t) # [1, n_actions, dim]
            xi_s = pred_pivot[0, pivot_action, :] # [dim]
            
            # Averaged Target
            # a_hat = argmax_a' value(E[xi(s', a')]) under selected reward model
            
            # Forward pass all next states [N, A, D]
            pred_next_all = model(c_nexts) 
            
            # Compute Mean Feature Vector over batch [A, D]
            xi_next_mean = torch.mean(pred_next_all, dim=0) 
            
            # Select action that maximizes value on the mean successor descriptor.
            q_next_mean = self.score_successor(
                xi_next_mean,
                task_index=task_i,
                w=self.fit_w[task_i] if self.reward_model == 'linear' else None,
            )
            if isinstance(q_next_mean, torch.Tensor):
                hat_a = int(torch.argmax(q_next_mean).item())
            else:
                hat_a = int(np.argmax(q_next_mean))
            
            # Gather the specific SFs for that chosen action from the full batch
            indices = torch.arange(c_nexts.shape[0], device=self.device)
            xi_next_selected = pred_next_all[indices, hat_a, :] # [N, dim]
            
            # Element-wise target: phi_p + gamma_p * xi(s'_p, hat{a})
            targets_N = c_phis + c_gammas * xi_next_selected
            
            # Average the calculated targets
            target_bar = torch.mean(targets_N, dim=0)
            
            diff = target_bar - xi_s
            loss = 0.5 * (diff.pow(2)).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()