import os
import torch
import numpy as np

def save_agent_weights(agent, agent_name, trial_id, root="weights"):
    """
    Saves weights for DQN, SFDQN, or FGSFDQN agents.
    Specific support for DeepSF/DeepFGSF 'psi' structure.
    """
    path = os.path.join(root, agent_name)
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"trial_{trial_id}.pt")
    
    payload = {
        "agent_name": agent_name,
        "trial_id": trial_id,
        "algorithm": getattr(agent, "algorithm", "standard")
    }

    # DQN Handling
    q_net = getattr(agent, "q", getattr(agent, "Q", None))
    if q_net is not None and isinstance(q_net, torch.nn.Module):
        payload["q_network"] = q_net.state_dict()
        
    # SF / FGSF Handling
    if hasattr(agent, "sf"):
        sf_module = agent.sf
        
        # Save 'fit_w' (List of weights for all tasks)
        if hasattr(sf_module, "fit_w"):
            payload["reward_weights_fit"] = sf_module.fit_w 
        # Fallback to single 'w'
        elif hasattr(sf_module, "w"):
            w = sf_module.w
            if torch.is_tensor(w):
                payload["reward_weights"] = w.detach().cpu().numpy()
            else:
                payload["reward_weights"] = w

        # Save PSI (Successor Feature Networks)
        if hasattr(sf_module, "psi"):
            payload["psi_networks"] = {}
            for i, (model, target_model, optimizer) in enumerate(sf_module.psi):
                payload["psi_networks"][i] = {
                    "model": model.state_dict(),
                    "target": target_model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }

        # Save extra state for nonlinear/SFR reward models.
        payload["sf_meta"] = {
            "reward_model": getattr(sf_module, "reward_model", None),
            "successor_representation": getattr(sf_module, "successor_representation", None),
            "reward_support": [np.asarray(x).tolist() for x in getattr(sf_module, "reward_support", [])],
            "sfr_centers": [np.asarray(x).tolist() for x in getattr(sf_module, "sfr_centers", [])],
            "sfr_center_reward_sums": list(getattr(sf_module, "sfr_center_reward_sums", [])),
            "sfr_center_reward_counts": list(getattr(sf_module, "sfr_center_reward_counts", [])),
        }
        if hasattr(sf_module, "reward_models"):
            payload["reward_models"] = {}
            for i, rm in enumerate(sf_module.reward_models):
                if isinstance(rm, torch.nn.Module):
                    payload["reward_models"][i] = rm.state_dict()
        if hasattr(sf_module, "reward_optimizers"):
            payload["reward_optimizers"] = {}
            for i, ropt in enumerate(sf_module.reward_optimizers):
                if ropt is not None:
                    payload["reward_optimizers"][i] = ropt.state_dict()

    torch.save(payload, filename)
    print(f"Saved: {filename}")
    
    
def load_agent_weights(agent, filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    checkpoint = torch.load(filename,weights_only=False)
    print(f"Loading {checkpoint.get('agent_name', 'Agent')} (Trial {checkpoint.get('trial_id', '?')})...")

    # Load SF / FGSF
    if hasattr(agent, "sf") and "psi_networks" in checkpoint:
        sf_module = agent.sf
        psi_data = checkpoint["psi_networks"]
        
        assert len(sf_module.psi) == len(psi_data), (
            f"Shape Mismatch: Agent has {len(sf_module.psi)} policies initialized, "
            f"but checkpoint contains {len(psi_data)}. "
            "Initialize tasks (agent.add_task) before loading."
        )

        for i, data in psi_data.items():
            idx = int(i)
            if idx < len(sf_module.psi):
                model, target, opt = sf_module.psi[idx]
                model.load_state_dict(data["model"])
                target.load_state_dict(data["target"])
                opt.load_state_dict(data["optimizer"])
    
    # Load Rewards
    if hasattr(agent, "sf"):
        if "reward_weights_fit" in checkpoint:
            # Restore full list of task weights
            agent.sf.fit_w = checkpoint["reward_weights_fit"]
        elif "reward_weights" in checkpoint:
            # Fallback for single task weight
            w_data = checkpoint["reward_weights"]
            if hasattr(agent.sf, "w") and torch.is_tensor(agent.sf.w):
                agent.sf.w.data = torch.from_numpy(w_data).to(agent.sf.w.device)
            else:
                agent.sf.w = w_data

        # Restore SFR/nonlinear reward-model state when available.
        sf_meta = checkpoint.get("sf_meta", {})
        if "reward_support" in sf_meta:
            agent.sf.reward_support = [np.asarray(x, dtype=np.float32) for x in sf_meta["reward_support"]]
        if "sfr_centers" in sf_meta:
            agent.sf.sfr_centers = [np.asarray(x, dtype=np.float32) for x in sf_meta["sfr_centers"]]
        if "sfr_center_reward_sums" in sf_meta:
            agent.sf.sfr_center_reward_sums = [float(x) for x in sf_meta["sfr_center_reward_sums"]]
        if "sfr_center_reward_counts" in sf_meta:
            agent.sf.sfr_center_reward_counts = [int(x) for x in sf_meta["sfr_center_reward_counts"]]

        reward_models = checkpoint.get("reward_models")
        if reward_models is not None and hasattr(agent.sf, "reward_models"):
            for i, sd in reward_models.items():
                idx = int(i)
                if idx < len(agent.sf.reward_models) and isinstance(agent.sf.reward_models[idx], torch.nn.Module):
                    agent.sf.reward_models[idx].load_state_dict(sd)
        elif getattr(agent.sf, "reward_model", "linear") == "nonlinear":
            print(
                "Warning: nonlinear reward checkpoint missing reward model parameters. "
                "This checkpoint likely predates SFR reward-model serialization and may evaluate poorly."
            )

        reward_opts = checkpoint.get("reward_optimizers")
        if reward_opts is not None and hasattr(agent.sf, "reward_optimizers"):
            for i, sd in reward_opts.items():
                idx = int(i)
                if idx < len(agent.sf.reward_optimizers) and agent.sf.reward_optimizers[idx] is not None:
                    agent.sf.reward_optimizers[idx].load_state_dict(sd)

    # Load DQN
    if "q_network" in checkpoint:
        if hasattr(agent, "q") and isinstance(agent.q, torch.nn.Module):
            agent.q.load_state_dict(checkpoint["q_network"])
        elif hasattr(agent, "Q") and isinstance(agent.Q, torch.nn.Module):
            agent.Q.load_state_dict(checkpoint["q_network"])

    print("Weights loaded successfully.")