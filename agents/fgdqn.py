import numpy as np
import random
import torch
import torch.nn.functional as F
from agents.agent import Agent

class FGDQN(Agent):
    
    def __init__(self, model_lambda, buffer, *args, test_epsilon=0.03, **kwargs):
        """
        FGDQN: Functional Gradient DQN.
        Uses a single network and allows gradients to flow through the target value 
        (minimizing the Bellman Residual).
        """
        super(FGDQN, self).__init__(*args, **kwargs)
        self.model_lambda = model_lambda
        self.buffer = buffer
        self.test_epsilon = test_epsilon
    
    def reset(self):
        Agent.reset(self)
        self.Q = self.model_lambda()
        self.buffer.reset()
        
    def get_Q_values(self, s, s_enc):
        self.Q.eval() # Switch to eval mode for inference
        with torch.no_grad():
            s_enc_tensor = torch.from_numpy(s_enc).float()

            if s_enc_tensor.ndim == 1:
                s_enc_tensor = s_enc_tensor.unsqueeze(0)
                
            device = next(self.Q.parameters()).device
            s_enc_tensor = s_enc_tensor.to(device)
            
            q_values = self.Q(s_enc_tensor)
            
        return q_values.cpu().numpy().flatten()

    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        self.Q.train() # Switch to train mode
        
        # Store experience
        self.buffer.append(s_enc, a, r, s1_enc, gamma)

        # Sample batch
        batch = self.buffer.replay()
        if batch is None:
            return
        
        states, actions, rewards, next_states, gammas = batch

        # Prepare Tensors
        device = next(self.Q.parameters()).device
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().flatten().to(device)
        rewards = torch.from_numpy(rewards).float().flatten().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        gammas = torch.from_numpy(gammas).float().flatten().to(device)

        # Main FGDQN Update
        self.Q.optimizer.zero_grad()

        # Compute Q(s, a)
        q_values = self.Q(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Keep gradients through next-state values.
        q_values_next = self.Q(next_states)
        max_next_q, _ = q_values_next.max(dim=1)

        # Construct Target
        # Minimize (Q(s,a) - Target)^2
        target_q = rewards + gammas * max_next_q

        # Compute Loss and Update
        loss = F.mse_loss(q_selected, target_q)
        loss.backward()
        self.Q.optimizer.step()
    
    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None, test_tasks=[], n_test_ev=1000):
        if viewers is None: 
            viewers = [None] * len(train_tasks)
        
        self.reset()
        for train_task in train_tasks:
            self.add_training_task(train_task)
            
        return_data = []
        for index, (train_task, viewer) in enumerate(zip(train_tasks, viewers)):
            self.set_active_training_task(index)
            for t in range(n_samples):
                self.next_sample(viewer, n_view_ev)

                if t % n_test_ev == 0:
                    Rs = [self.test_agent(task) for task in test_tasks]
                    if len(Rs) > 0:
                        avg_R = np.mean(np.array(Rs))
                        return_data.append(avg_R)
                        print(f'Test performance: {avg_R:.4f}')
        return return_data
    
    def get_test_action(self, s_enc):
        if random.random() <= self.test_epsilon:
            a = random.randrange(self.n_actions)
        else:
            q = self.get_Q_values(s_enc, s_enc)
            a = np.argmax(q)
        return a

    def test_agent(self, task, return_history=False, visualize=False, pause=0.12, max_steps=None):
        T = max_steps if max_steps is not None else self.T
        total_reward = 0.0
        s = task.initialize()
        s_enc = self.encoding(s)

        states = [s]     
        actions = []
        rewards = []

        for step in range(T):
            a = self.get_test_action(s_enc)
            s1, r, done = task.transition(a)

            actions.append(a)
            rewards.append(r)
            total_reward += r

            states.append(s1)
            s_enc = self.encoding(s1)
            s = s1

            if done:
                break

        episode = {
            "total_reward": total_reward,
            "steps": len(actions),
            "states": states,
            "actions": actions,
            "rewards": rewards
        }

        if visualize:
            Agent.render_episode_history_rich(episode, task, agent=self, pause=pause)

        if return_history:
            return episode
        return total_reward