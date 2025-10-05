# Comparative Analysis of Offline RL Algorithms: BC, CQL, IQL, and Enhanced TD3+BC

This repository contains the code for an empirical study of several state-of-the-art offline reinforcement learning (Offline-RL) algorithms. The project evaluates and compares the performance of Behavioral Cloning (BC), Twin Delayed Deep Deterministic Policy Gradient with Behavioral Cloning (TD3+BC), Conservative Q-Learning (CQL), and Implicit Q-Learning (IQL).

The implementations are tested on continuous control environments from the D4RL/Minari benchmarks, using datasets of varying quality ('medium' and 'expert').

## Project Description

Offline Reinforcement Learning is a paradigm where an agent learns from a fixed, pre-collected dataset of experiences without further interaction with the environment. This approach is crucial for real-world applications where online data collection is costly, time-consuming, or unsafe. However, it introduces challenges like distributional shift, where the learned policy may query out-of-distribution (OOD) actions, leading to overestimation and unstable learning.

This project investigates these challenges by evaluating four prominent Offline-RL algorithms across eight continuous control environments: HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Pusher, Reacher, Swimmer, and Walker2D. A key contribution is a custom implementation of TD3+BC with several enhancements designed to improve stability and performance, especially on lower-quality datasets.

## Algorithms Implemented

### 1. Behavioral Cloning (BC)

-   **Description**: BC treats offline RL as a supervised learning problem. It directly imitates the actions in the dataset by training a policy to minimize the difference between its predicted actions and the actions from the dataset (e.g., using Mean Squared Error). It is simple and avoids value estimation but cannot outperform the best policy in the dataset.
-   **Implementation**: `behavior cloning/bc.py`
    -   A small feed-forward neural network (`BCPolicySmall`) maps observations to actions.
    -   The policy is trained using the Adam optimizer and MSE loss.
    -   The training loop includes per-epoch evaluation to track performance over time.

### 2. Conservative Q-Learning (CQL) & Implicit Q-Learning (IQL)

-   **Description**:
    -   **CQL**: Learns a conservative Q-function by adding a regularizer to the standard Bellman error objective. This regularizer penalizes Q-values for unseen actions, preventing overestimation of out-of-distribution actions and ensuring the learned policy stays close to the data distribution.
    -   **IQL**: Avoids explicit policy constraints by using expectile regression to learn a value function and advantage-weighted behavioral cloning to extract a policy. This allows for stable offline learning without directly evaluating OOD Q-values.
-   **Implementation**: `CQL_IQL/cql_iql.py`
    -   Uses the `d3rlpy` library for the core CQL and IQL algorithm implementations.
    -   Includes dynamic hyperparameter selection (`choose_profile_cql`, `choose_profile_iql`) based on dataset quality (medium or expert).
    -   Applies episode filtering for medium-quality datasets, keeping only the top-performing trajectories to stabilize training.
    -   The training process is parallelized to run multiple seeds efficiently.

### 3. TD3+BC (Twin Delayed Deep Deterministic Policy Gradient + Behavioral Cloning)

This project includes two versions of TD3+BC: a default implementation using `d3rlpy` and a custom implementation with several enhancements.

#### Default TD3+BC

-   **Description**: This is a standard implementation of TD3+BC, which combines the TD3 actor-critic algorithm with a behavioral cloning term. The policy is regularized to stay close to the behavior policy from the dataset, mitigating distributional shift.
-   **Implementation**: `TD3PlusBC/default.py`
    -   Uses the `d3rlpy` library's `TD3PlusBCConfig` with a fixed set of hyperparameters.
    -   Provides a baseline for comparison against the custom implementation.

#### Custom TD3+BC

-   **Description**: This version introduces several modifications to improve performance and robustness, particularly on medium-quality datasets.
-   **Implementation**: `TD3PlusBC/custom_TD3PlusBC.py`

Our implementation introduces several modifications that vary based on dataset quality.

##### Dynamic Hyperparameter Selection
The `choose_profile()` function selects hyperparameters based on dataset name matching. Expert and medium datasets receive distinct configurations for batch size, BC weight, policy noise, and lambda clipping bounds.

```python
# Default configuration inherited by all profiles
cfg = dict(
    label="DEFAULT", alpha=1.5, bc_weight=2.0, bc_warmup=0,
    policy_noise=0.08 * max_action, noise_clip=0.25 * max_action,
    policy_freq=2, episode_filter_frac=0.0,
    target=900.0, batch_size=256, lambda_clip=(0.0, 10.0),
    discount=0.99, tau=0.005
)

if "expert" in n:
    cfg.update(dict(
        label="EXPERT", alpha=2.5, bc_weight=1.0,
        policy_noise=0.10 * max_action, 
        noise_clip=0.30 * max_action,
        target=1000.0, batch_size=256
    ))
elif "medium" in n:
    cfg.update(dict(
        label="MEDIUM", alpha=1.0, bc_weight=4.0,
        lambda_clip=(0.0, 1.2),
        policy_noise=0.20 * max_action,
        noise_clip=0.50 * max_action,
        batch_size=1024,
        episode_filter_frac=0.50,
        target=1000.0
    ))
```

##### BC Weight Annealing
The BC weight anneals linearly during the first 5000 training steps to 50% of its initial value, then remains constant. While the code supports an optional warmup phase for pure behavioral cloning, all profiles set `bc_warmup=0`, so the actor loss includes both BC and Q-maximization terms from the first update.

```python
progress = min(1.0, self.total_it / 5000.0)
bc_w = self.bc_weight * (1.0 - 0.5 * progress)

if self.total_it <= self.bc_warmup:  # Never true when bc_warmup=0
    actor_loss = bc_w * bc_mse
else:  # Always executed in our experiments
    Q = self.critic.Q1(s, pi)
    lam = self.alpha / Q.abs().mean().detach()
    lam = torch.clamp(lam, self.lambda_clip[0], self.lambda_clip[1])
    actor_loss = -lam * Q.mean() + bc_w * bc_mse
```

##### Explicit State Normalization
State observations are normalized using dataset-wide mean and standard deviation.

```python
def normalize_states(self, eps=1e-6):
    mean = self.state[:self.size].mean(0, keepdims=True)
    std = self.state[:self.size].std(0, keepdims=True) + eps
    self.state[:self.size] = (self.state[:self.size] - mean) / std
    self.next_state[:self.size] = (self.next_state[:self.size] - mean) / std
    return mean, std
```

##### Episode Filtering for Medium Datasets
Episodes are filtered by cumulative return using quantile thresholding. Medium datasets retain the top 50% of episodes, while expert datasets use no filtering.

```python
def filter_topk_by_return(ds, keep_frac=0.8):
    eps = list(ds.iterate_episodes())
    returns = np.array([ep.rewards.sum() for ep in eps])
    thresh = np.quantile(returns, 1.0 - keep_frac)
    keep_eps = [ep for ep, R in zip(eps, returns) if R >= thresh]
    return _d4rl_from_episodes(keep_eps)

# In train_agent(): medium datasets use keep_frac=0.50
if cfg.get("episode_filter_frac", 0.0) > 0:
    dset, _, _ = filter_topk_by_return(ds, keep_frac=cfg["episode_filter_frac"])
```

## How to Run the Code

Each algorithm is implemented in a self-contained script.

### 1. Behavioral Cloning

To run the BC experiment:
```bash
python "behavior cloning/bc.py"
```
The script will train the BC policy on the specified datasets, run evaluations, and generate plots for training loss and evaluation rewards.

### 2. CQL and IQL

To run the CQL and IQL experiments:
```bash
python CQL_IQL/cql_iql.py
```
This script will train both CQL and IQL agents in parallel across multiple seeds and datasets. It will save performance plots to the `CQL_IQL/plots/` directory.

### 3. TD3+BC

-   **To run the default TD3+BC implementation:**
    ```bash
    python TD3PlusBC/default.py
    ```
-   **To run the custom TD3+BC implementation:**
    ```bash
    python TD3PlusBC/custom_TD3PlusBC.py
    ```
Both scripts will train the respective agents, evaluate their performance, and generate summary plots and tables.

## Experimental Results Summary

As detailed in the accompanying paper, the experimental results show:

-   **BC** is a strong baseline on **expert** datasets, where pure imitation is sufficient. However, its performance degrades significantly on **medium** datasets containing suboptimal behavior.
-   The **custom TD3+BC** substantially outperforms the default implementation and the BC baseline on **medium** datasets. The enhancements (dynamic hyperparameters, BC weight annealing, state normalization, and episode filtering) prove effective at improving learning from noisy, mixed-quality data.
-   **IQL and CQL** demonstrate strong and stable performance, particularly on medium-quality datasets where they can effectively identify and leverage high-quality behaviors.
-   The findings highlight the trade-off between imitation and reinforcement learning. While imitation provides a stable foundation, value-based updates are crucial for improving upon suboptimal data.

## Reference Paper

For a detailed analysis, methodology, and results, please refer to the research paper:

**"Offline-RL: A Comprehensive Evaluation of State-of-the-Art Algorithms"**
