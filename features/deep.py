
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from features.successor import SF

class SFNetwork1(nn.Module):
    def __init__(self, input_dim, n_actions, n_features):
        super(SFNetwork1, self).__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.n_features = n_features
        # Simple MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, n_actions * n_features)
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, np.sqrt(1.0 / m.weight.shape[1]))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.net(x)
        # Reshape to [batch, n_actions, n_features]
        return out.view(-1, self.n_actions, self.n_features)

class SFNetwork2(nn.Module):
    def __init__(self, input_dim, n_actions, n_features, hidden_dims=[256, 256]):
        super(SFNetwork2, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features

        layers = []
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU()) 
            input_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)

        self.output_layer = nn.Linear(input_dim, n_actions * n_features)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, np.sqrt(1.0 / m.weight.shape[1]))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        return x.view(-1, self.n_actions, self.n_features)

class DeepSF(SF):
    """Successor features with PyTorch function approximation."""
    def __init__(self, input_dim, n_actions, n_features=None,
                 learning_rate=1e-3, learning_rate_w=0.5,
                 target_update_ev=1000, device=None, *args, **kwargs):
        super(DeepSF, self).__init__(learning_rate_w, *args, **kwargs)

        self.input_dim = input_dim
        self.n_actions = n_actions
        self.n_features = n_features if n_features is not None else n_actions
        self.learning_rate = learning_rate               # network LR
        self.target_update_ev = target_update_ev
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.updates_since_target_updated = []
        self.reset()

    def reset(self):
        SF.reset(self)
        self.updates_since_target_updated = []

    def build_successor(self, task, source=None):
        if self.n_tasks == 0:
            self.n_actions = task.action_count()
            self.n_features = task.feature_dim()
            self.input_dim = task.encode_dim()

        model = None
        target_model = None
        
        model = SFNetwork2(self.input_dim, self.n_actions, self.n_features).to(self.device)
        target_model = SFNetwork2(self.input_dim, self.n_actions, self.n_features).to(self.device)
        target_model.load_state_dict(model.state_dict())
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        if source is not None and self.n_tasks > 0:
            source_psi, _, _ = self.psi[source]
            model.load_state_dict(source_psi.state_dict())
            target_model.load_state_dict(source_psi.state_dict())

        self.updates_since_target_updated.append(0)
        return (model, target_model, optimizer)

    def get_successor(self, state, policy_index):
        model, _, _ = self.psi[policy_index]
        model.eval()
        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                state_tensor = state.float().to(self.device)
                if state_tensor.ndim == 1:
                    state_tensor = state_tensor.unsqueeze(0)
            else:
                arr = np.asarray(state)
                if arr.ndim == 1:
                    arr = arr[None, ...]
                state_tensor = torch.from_numpy(arr).float().to(self.device)
            psi = model(state_tensor)  # torch tensor [B, A, F]
        if isinstance(state, torch.Tensor):
            return psi
        return psi.cpu().numpy()

    def get_successors(self, state):
        input_is_torch = isinstance(state, torch.Tensor)
        if input_is_torch:
            state_tensor = state.float().to(self.device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
        else:
            arr = np.asarray(state)
            if arr.ndim == 1:
                arr = arr[None, ...]
            state_tensor = torch.from_numpy(arr).float().to(self.device)
        all_psi = []
        for model, _, _ in self.psi:
            model.eval()
            with torch.no_grad():
                psi = model(state_tensor)  # torch [B, A, F]
                all_psi.append(psi)
        # all_psi: list of length n_tasks; each element [B, A, F]
        all_psi = torch.stack(all_psi, dim=0)          # [n_tasks, B, A, F]
        all_psi = all_psi.permute(1, 0, 2, 3)          # [B, n_tasks, A, F]
        if not input_is_torch:
            return all_psi.detach().cpu().numpy()
        return all_psi


    def update_successor(self, transitions, policy_index):
        if transitions is None:
            return
        model, target_model, optimizer = self.psi[policy_index]
        model.train()
        target_model.eval()
        states, actions, phis, next_states, gammas = transitions

        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states).float().to(self.device)
        else:
            states = states.float().to(self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(actions).long().to(self.device)
        else:
            actions = actions.long().to(self.device)
        if not isinstance(phis, torch.Tensor):
            phis = torch.from_numpy(phis).float().to(self.device)
        else:
            phis = phis.float().to(self.device)
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.from_numpy(next_states).float().to(self.device)
        else:
            next_states = next_states.float().to(self.device)
        if not isinstance(gammas, torch.Tensor):
            gammas = torch.from_numpy(gammas).float().to(self.device).view(-1, 1)
        else:
            gammas = gammas.float().to(self.device).view(-1, 1)

        n_batch = states.shape[0]
        device = self.device
        indices_t = torch.arange(n_batch, device=device, dtype=torch.long)


        # Next actions come from GPI, kept on torch/device.
        q1, _ = self.GPI(next_states, policy_index)
        if isinstance(q1, torch.Tensor):
            next_actions = torch.argmax(torch.max(q1, dim=1).values, dim=-1).long().to(device)
        else:
            next_actions_np = np.argmax(np.max(q1, axis=1), axis=-1)
            next_actions = torch.from_numpy(next_actions_np).long().to(device)


        # Compute targets
        with torch.no_grad():
            target_out = target_model(next_states)  # [B, A, F]
            target_psi_next = target_out[indices_t, next_actions, :]  # [B, F]

        # Forward pass
        phis = phis.view(n_batch, -1)  
        targets = phis + gammas * target_psi_next
        
        psi_pred = model(states)
        psi_pred_actions = psi_pred[indices_t, actions, :]

        # Loss: MSE between predicted and target
        loss_fn = nn.MSELoss()
        loss = loss_fn(psi_pred_actions, targets)

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Update target network
        self.updates_since_target_updated[policy_index] += 1
        if self.updates_since_target_updated[policy_index] >= self.target_update_ev:
            target_model.load_state_dict(model.state_dict())
            self.updates_since_target_updated[policy_index] = 0
