import torch
import torch.nn as nn
import numpy as np
from features.deep import DeepSF

class DeepFGSF(DeepSF):
    """
    Full-gradient successor features (FG-SFRQL).

    One update on block k minimizes the (averaged) squared SF Bellman residual
        || (1/N) sum_p [phi_p + gamma_p * xi_k(s'_p, b_p)] - xi_k(s, a) ||^2
    and differentiates through both the prediction xi_k(s, a) and the bootstrapped
    target xi_k(s'_p, b_p). N = 1 gives the single-sample update of Alg. 1/2, N > 1
    the averaged update of Alg. 3/4.

    Selections follow the paper:
      - the GPI prior c is chosen at the pivot state s under the current task's reward;
      - the current task i bootstraps with the GPI action at each s'_p (max over all policies);
      - a prior c != i bootstraps with its own greedy action under its own reward;
      - greedy actions are chosen separately for every next-state sample.
    All selections are computed from the parameters before the update.
    """
    def __init__(self, learning_rate_prior=None, *args, **kwargs):
        super(DeepFGSF, self).__init__(*args, **kwargs)
        self.learning_rate_prior = learning_rate_prior if learning_rate_prior is not None else self.learning_rate

    def _set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _to_tensor(self, x, dtype=torch.float):
        if not isinstance(x, torch.Tensor):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)
        return x.to(dtype=dtype, device=self.device)

    def _reward_w(self, task_index):
        return self.fit_w[task_index] if self.reward_model == 'linear' else None

    # ===========================================================================
    # SELECTIONS (no gradient)
    # ===========================================================================
    def gpi_prior(self, states, task_index):
        """GPI prior c(s) = argmax_j max_a Q^i_j(s, a) at the pivot states: [M]."""
        q = self._q_all_policies(states, task_index)
        return q.max(dim=2).values.argmax(dim=1)

    def gpi_actions(self, next_states, task_index):
        """GPI target action argmax_a' max_j Q^i_j(s', a') for the current task: [M]."""
        q = self._q_all_policies(next_states, task_index)
        return q.max(dim=1).values.argmax(dim=-1)

    def own_greedy_actions(self, next_states, policy_index):
        """Greedy target action of policy c under its own reward: [M]."""
        model, _, _ = self.psi[policy_index]
        model.eval()
        with torch.no_grad():
            psi = model(next_states)  # [M, A, F]
            q = self.score_successor(psi, task_index=policy_index, w=self._reward_w(policy_index))
        q = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        return q.argmax(dim=-1)

    # ===========================================================================
    # FULL-GRADIENT UPDATES
    # ===========================================================================
    def _fg_step(self, k, lr, states, actions, phis, next_states, gammas, next_actions):
        """
        One full-gradient step on block k.
        states [B, S], actions [B], phis [B, N, F], next_states [B, N, S],
        gammas [B, N, 1], next_actions [B, N].
        """
        model, _, optimizer = self.psi[k]
        self._set_lr(optimizer, lr)
        model.train()
        optimizer.zero_grad()

        B, N, S = next_states.shape
        rows = torch.arange(B, device=self.device)
        xi_s = model(states)[rows, actions, :]  # [B, F]

        # Gradients flow through the bootstrapped target as well (full gradient).
        pred_next = model(next_states.reshape(B * N, S)).view(B, N, self.n_actions, -1)
        idx = next_actions.view(B, N, 1, 1).expand(B, N, 1, pred_next.shape[-1])
        xi_next = pred_next.gather(2, idx).squeeze(2)  # [B, N, F]

        target_bar = (phis + gammas * xi_next).mean(dim=1)  # [B, F]
        loss = 0.5 * (target_bar - xi_s).pow(2).mean()
        loss.backward()
        if getattr(self, 'grad_clip', 1.0) > 0:  # grad_clip <= 0 disables clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(self, 'grad_clip', 1.0))
        optimizer.step()
        return float(loss.item())

    def update_full_gradient(self, batch, task_i):
        """
        Full-gradient update for current task task_i and the GPI-selected priors.
        batch = (states [B, S], actions [B], phis [B, N, F] or [B, F],
                 next_states [B, N, S] or [B, S], gammas [B, N] or [B]).
        """
        states, actions, phis, next_states, gammas = self._prepare_batch(batch)
        B, N, S = next_states.shape
        flat_next = next_states.reshape(B * N, S)

        # GPI prior per pivot: at the pivot state s (paper Alg. 1, default) or at the next state s'
        # (prior_state = 'next', the behaviour of the original code release).
        if getattr(self, 'prior_state', 's') == 'next':
            prior = self.gpi_prior(next_states[:, 0], task_i)
        else:
            prior = self.gpi_prior(states, task_i)                  # [B]

        if getattr(self, 'update_mode', 'joint') == 'grouped':
            # Original code release: one step per prior group; the current task is stepped on every
            # group's sub-batch (so up to n_groups steps per call), and the prior c on its own group.
            for c in torch.unique(prior).tolist():
                mask = prior == c
                sub = (states[mask], actions[mask], phis[mask], next_states[mask], gammas[mask])
                flat = sub[3].reshape(-1, S)
                self._fg_step(task_i, self.learning_rate, *sub, self.gpi_actions(flat, task_i).view(-1, N))
                if c != task_i:
                    self._fg_step(c, self.learning_rate_prior, *sub, self.own_greedy_actions(flat, c).view(-1, N))
            return

        # Selections with the pre-update parameters.
        b_i = self.gpi_actions(flat_next, task_i).view(B, N)       # [B, N]
        prior_targets = {}
        for c in torch.unique(prior).tolist():
            if c == task_i:
                continue
            mask = prior == c
            b_c = self.own_greedy_actions(next_states[mask].reshape(-1, S), c).view(-1, N)
            prior_targets[c] = (mask, b_c)

        # Current task: every pivot in the batch.
        self._fg_step(task_i, self.learning_rate, states, actions, phis, next_states, gammas, b_i)

        # Priors: only the pivots where GPI selects them.
        for c, (mask, b_c) in prior_targets.items():
            self._fg_step(c, self.learning_rate_prior, states[mask], actions[mask],
                          phis[mask], next_states[mask], gammas[mask], b_c)

    def update_single_sample(self, transition, task_i, task_c=None):
        """Single-sample (N = 1) full-gradient update; the priors are selected internally."""
        self.update_full_gradient(transition, task_i)

    def update_averaged_batch(self, batch_data, task_i):
        """
        Averaged full-gradient update.
        batch_data: (p_states [B, S], p_actions [B], c_phis [B, N, F],
                     c_nexts [B, N, S], c_gammas [B, N])
        """
        self.update_full_gradient(batch_data, task_i)

    def update_averaged(self, pivot_state, pivot_action, conditional_batch, task_i, task_c=None, hat_a=None):
        """Averaged update for a single pivot (kept for backward compatibility)."""
        _, _, c_phis, c_next_states, c_gammas = conditional_batch
        batch = (
            np.asarray(pivot_state, dtype=np.float32).reshape(1, -1),
            np.asarray([int(pivot_action)]),
            np.asarray(c_phis, dtype=np.float32)[None, ...],
            np.asarray(c_next_states, dtype=np.float32)[None, ...],
            np.asarray(c_gammas, dtype=np.float32).reshape(1, -1),
        )
        self.update_full_gradient(batch, task_i)
