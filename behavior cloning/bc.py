import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import minari
import random

# ---------------- Small Policy Network ----------------
class BCPolicySmall(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )
    def forward(self, x):
        return self.net(x)


# ---------------- Training Loop with per-epoch evaluation ----------------
def train_bc(dataset_name, num_epochs, batch_size, learning_rate, seed):
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load dataset and env
    dataset = minari.load_dataset(dataset_name, download=True)
    env = dataset.recover_environment()
    env.reset(seed=seed)
    if hasattr(env, "action_space"):
        env.action_space.seed(seed)
    if hasattr(env, "observation_space"):
        env.observation_space.seed(seed)

    # Prepare offline data
    episodes = dataset.sample_episodes(dataset.total_episodes)
    obs_list, act_list = [], []
    for ep in episodes:
        obs_list.append(ep.observations[:-1])
        act_list.append(ep.actions)
    observations = torch.tensor(np.concatenate(obs_list, axis=0), dtype=torch.float32)
    actions = torch.tensor(np.concatenate(act_list, axis=0), dtype=torch.float32)
    tensor_dataset = TensorDataset(observations, actions)
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

    # Init policy + optimizer
    policy = BCPolicySmall(obs_dim=observations.shape[1], act_dim=actions.shape[1])
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Logs
    loss_history = []
    reward_history = []

    for epoch in range(num_epochs):
        # ---- Train for one epoch ----
        epoch_loss = 0
        for obs_batch, act_batch in loader:
            pred = policy(obs_batch)
            loss = loss_fn(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * obs_batch.size(0)
        epoch_loss /= len(tensor_dataset)
        loss_history.append(epoch_loss)

        # ---- Evaluate after each epoch ----
        obs, _ = env.reset(seed=seed + epoch)  # vary seed per eval
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        reward_history.append(total_reward)

    return loss_history, reward_history


# ---------------- Main Experiment ----------------
if __name__ == "__main__":
    NUM_RUNS = 10         # fewer runs for speed
    NUM_EPOCHS = 20      # you can increase later
    BATCH_SIZE = 64
    LR = 1e-3
    DATASET_NAMES = [
        "mujoco/reacher/medium-v0",
    ]

    all_results = {}

    for dataset in DATASET_NAMES:
        print(f"\n{'='*25} Running Experiment on: {dataset} {'='*25}")
        run_rewards, run_losses = [], []

        for i in range(NUM_RUNS):
            current_seed = np.random.randint(0, 1_000_000)
            print(f"--- Run {i+1}/{NUM_RUNS}, Seed {current_seed} ---")
            losses, rewards = train_bc(dataset, NUM_EPOCHS, BATCH_SIZE, LR, current_seed)
            run_rewards.append(rewards)
            run_losses.append(losses)

        run_rewards = np.array(run_rewards)
        run_losses = np.array(run_losses)

        all_results[dataset] = {
            "losses_all_runs": run_losses,
            "rewards_all_runs": run_rewards,
            "loss_mean": np.mean(run_losses, axis=0),
            "loss_std": np.std(run_losses, axis=0),
            "reward_mean_curve": np.mean(run_rewards, axis=0),
            "reward_std_curve": np.std(run_rewards, axis=0),
            "reward_mean_final": np.mean(run_rewards[:, -1]),
            "reward_std_final": np.std(run_rewards[:, -1]),
        }

    # ---------------- Plot Results ----------------
    print("\n" + "="*30 + " Plotting Results " + "="*30)
    for dataset, results in all_results.items():
        epochs = np.arange(1, NUM_EPOCHS + 1)

        # ---- Plot training loss ----
        plt.figure(figsize=(10, 6))
        mean, std = results["loss_mean"], results["loss_std"]
        plt.plot(epochs, mean, label="Mean Training Loss", color="b", linewidth=2)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.2, color="b", label="Std. Dev.")
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("MSE Loss", fontsize=14)
        plt.title(f"BC Training Loss on {dataset}\n(Mean ± Std over {NUM_RUNS} runs)", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

        # ---- Plot evaluation reward ----
        plt.figure(figsize=(10, 6))
        mean, std = results["reward_mean_curve"], results["reward_std_curve"]
        plt.plot(epochs, mean, label="Mean Evaluation Reward", color="g", linewidth=2)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.2, color="g", label="Std. Dev.")
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Evaluation Return", fontsize=14)
        plt.title(f"BC Performance on {dataset}\n(Mean ± Std over {NUM_RUNS} runs)", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    # ---------------- Summary ----------------
    print("\n" + "="*25 + " Final Performance Summary " + "="*25)
    header = f"{'Dataset':<40} | Reward (Mean ± Std)"
    print(header); print("-" * len(header))
    for dataset, results in all_results.items():
        summary_str = f"{results['reward_mean_final']:.2f} ± {results['reward_std_final']:.2f}"
        print(f"{dataset:<40} | {summary_str}")
