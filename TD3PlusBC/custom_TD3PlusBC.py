
import copy, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from collections import deque
import minari
import matplotlib.pyplot as plt


BASE_SEED = 42
np.random.seed(BASE_SEED); random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed(BASE_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def dataset_to_d4rl(ds):
    obs_list, act_list, nxt_list, rew_list, term_list, tout_list = [], [], [], [], [], []
    for ep in ds.iterate_episodes():
        o = ep.observations.astype(np.float32)
        a = ep.actions.astype(np.float32)
        r = ep.rewards.astype(np.float32)
        T = len(a)
        o_t = o[:T]
        if len(o) >= T + 1:
            no_t = o[1:T+1]
        else:
            no_t = np.vstack([o[1:], o[-1:]]).astype(np.float32)
        t = np.zeros(T, dtype=np.float32)
        u = np.zeros(T, dtype=np.float32)
        if len(ep.terminations) > 0 and bool(ep.terminations[-1]): t[-1] = 1.0
        if len(ep.truncations) > 0 and bool(ep.truncations[-1]): u[-1] = 1.0
        obs_list.append(o_t); act_list.append(a); nxt_list.append(no_t)
        rew_list.append(r[:T]); term_list.append(t); tout_list.append(u)
    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)
    nxt = np.concatenate(nxt_list, axis=0)
    rew = np.concatenate(rew_list, axis=0)
    term = np.concatenate(term_list, axis=0)
    tout = np.concatenate(tout_list, axis=0)
    return {
        "observations": obs, "actions": act, "next_observations": nxt,
        "rewards": rew, "terminals": term, "timeouts": tout,
    }

def compute_stats_from_d4rl(d):
    return dict(
        transitions=len(d["rewards"]), rewards_mean=float(d["rewards"].mean()),
        rewards_std=float(d["rewards"].std()), terminals_frac=float(d["terminals"].mean()),
        timeouts_frac=float(d["timeouts"].mean()),
    )

def choose_profile(dataset_name: str, stats: dict, max_action: float):
    n = dataset_name.lower()
    cfg = dict(
        label="DEFAULT", alpha=1.5, bc_weight=2.0, bc_warmup=0,
        policy_noise=0.08 * max_action, noise_clip=0.25 * max_action,
        policy_freq=2, episode_filter_frac=0.0,
        target=900.0, batch_size=256, lambda_clip=(0.0, 10.0),
        discount=0.99, tau=0.005 # Added missing defaults
    )
    if "expert" in n:
        cfg.update(dict(
            label="EXPERT", alpha=2.5, bc_weight=1.0,
            policy_noise=0.10 * max_action, noise_clip=0.30 * max_action,
            target=1000.0, batch_size=256,
        ))
    elif "medium" in n:
        cfg.update(dict(
            label="MEDIUM", alpha=1.0, bc_weight=4.0,
            lambda_clip=(0.0, 1.2), policy_noise=0.20 * max_action,
            noise_clip=0.50 * max_action, batch_size=1024,
            episode_filter_frac=0.50, target=1000.0,
        ))
    else:
        if stats["rewards_mean"] > 0.999 and stats["terminals_frac"] < 0.005:
            cfg.update(dict(label="EXPERTLIKE", alpha=2.5, bc_weight=1.0))
        else:
            cfg.update(dict(label="MEDIUMLIKE", alpha=0.75, bc_weight=2.5, batch_size=1024))
    return cfg

def _d4rl_from_episodes(episodes):
    obs_list, act_list, nxt_list, rew_list, term_list, tout_list = [], [], [], [], [], []
    for ep in episodes:
        o, a, r = ep.observations.astype(np.float32), ep.actions.astype(np.float32), ep.rewards.astype(np.float32)
        T = len(a)
        o_t = o[:T]
        no_t = o[1:T+1] if len(o) >= T+1 else np.vstack([o[1:], o[-1:]]).astype(np.float32)
        t, u = np.zeros(T, np.float32), np.zeros(T, np.float32)
        if len(ep.terminations) and ep.terminations[-1]: t[-1] = 1.0
        if len(ep.truncations) and ep.truncations[-1]: u[-1] = 1.0
        obs_list.append(o_t); act_list.append(a); nxt_list.append(no_t)
        rew_list.append(r[:T]); term_list.append(t); tout_list.append(u)
    return {
        "observations": np.concatenate(obs_list), "actions": np.concatenate(act_list),
        "next_observations": np.concatenate(nxt_list), "rewards": np.concatenate(rew_list),
        "terminals": np.concatenate(term_list), "timeouts": np.concatenate(tout_list),
    }

def filter_topk_by_return(ds, keep_frac=0.8):
    eps = list(ds.iterate_episodes())
    returns = np.array([ep.rewards.sum() for ep in eps], dtype=np.float32)
    thresh = np.quantile(returns, 1.0 - keep_frac)
    print(f"  Filtering episodes with return < {thresh:.2f}")
    keep_eps = [ep for ep, R in zip(eps, returns) if R >= thresh]
    return _d4rl_from_episodes(keep_eps), len(keep_eps), len(eps)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(2e6)): 
        self.max_size, self.size = max_size, 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done_term_only = np.zeros((max_size, 1), dtype=np.float32)
        self.device = device
    def load_from_d4rl(self, d):
        s, a, ns = d["observations"], d["actions"], d["next_observations"]
        r, terminals = d["rewards"].reshape(-1, 1), d["terminals"].reshape(-1, 1)
        not_done_term_only = 1.0 - terminals
        n = s.shape[0]
        self.state[:n], self.action[:n], self.next_state[:n] = s, a, ns
        self.reward[:n], self.not_done_term_only[:n] = r, not_done_term_only
        self.size = n
    def normalize_states(self, eps=1e-6):
        mean = self.state[:self.size].mean(0, keepdims=True)
        std = self.state[:self.size].std(0, keepdims=True) + eps
        self.state[:self.size] = (self.state[:self.size] - mean) / std
        self.next_state[:self.size] = (self.next_state[:self.size] - mean) / std
        return mean, std
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.state[idx]).to(self.device),
            torch.from_numpy(self.action[idx]).to(self.device),
            torch.from_numpy(self.next_state[idx]).to(self.device),
            torch.from_numpy(self.reward[idx]).to(self.device),
            torch.from_numpy(self.not_done_term_only[idx]).to(self.device),
        )

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1_l1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_l2 = nn.Linear(256, 256)
        self.q1_l3 = nn.Linear(256, 1)
        self.q2_l1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_l2 = nn.Linear(256, 256)
        self.q2_l3 = nn.Linear(256, 1)
    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        q1 = F.relu(self.q1_l1(sa)); q1 = F.relu(self.q1_l2(q1)); q1 = self.q1_l3(q1)
        q2 = F.relu(self.q2_l1(sa)); q2 = F.relu(self.q2_l2(q2)); q2 = self.q2_l3(q2)
        return q1, q2
    def Q1(self, s, a):
        sa = torch.cat([s, a], dim=1)
        q1 = F.relu(self.q1_l1(sa)); q1 = F.relu(self.q1_l2(q1)); q1 = self.q1_l3(q1)
        return q1

class TD3_BC:
    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2, alpha=2.5,
                 bc_weight=1.0, bc_warmup=0, lambda_clip=(0.0, 10.0)):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.max_action, self.discount, self.tau = max_action, discount, tau
        self.policy_noise, self.noise_clip, self.policy_freq = policy_noise, noise_clip, policy_freq
        self.alpha, self.bc_weight, self.bc_warmup = alpha, bc_weight, bc_warmup
        self.lambda_clip = lambda_clip
        self.total_it = 0
    def select_action(self, state_np):
        with torch.no_grad():
            s = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            return self.actor(s).cpu().numpy().flatten()
    def train_step(self, rb, batch_size=512):
        self.total_it += 1
        s, a, ns, r, not_done = rb.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = (self.actor_target(ns) + noise).clamp(-self.max_action, self.max_action)
            tq1, tq2 = self.critic_target(ns, na)
            target = r + not_done * self.discount * torch.min(tq1, tq2)
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(s)
            bc_mse = F.mse_loss(pi, a)
            progress = min(1.0, self.total_it / 5000.0)
            bc_w = self.bc_weight * (1.0 - 0.5 * progress)
            if self.total_it <= self.bc_warmup:
                actor_loss = bc_w * bc_mse
            else:
                Q = self.critic.Q1(s, pi)
                lam = self.alpha / Q.abs().mean().detach()
                if self.lambda_clip is not None:
                    lam = torch.clamp(lam, self.lambda_clip[0], self.lambda_clip[1])
                actor_loss = -lam * Q.mean() + bc_w * bc_mse
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.mul_(1-self.tau); tp.data.add_(self.tau * p.data)
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.data.mul_(1-self.tau); tp.data.add_(self.tau * p.data)

def eval_policy(policy, env, mean, std, seed, episodes=10):
    policy.actor.eval()
    total_return = 0.0
    for i in range(episodes):
        try:
            state, _ = env.reset(seed=seed + 100 + i)
        except TypeError:
            random.seed(seed + 100 + i); np.random.seed(seed + 100 + i)
            state = env.reset()
        done, episode_return = False, 0.0
        while not done:
            state_norm = (np.array(state, dtype=np.float32) - mean) / std
            action = policy.select_action(state_norm)
            step_result = env.step(action)
            if len(step_result) == 5:
                state, reward, term, trunc, _ = step_result
                done = term or trunc
            else:
                state, reward, done, _ = step_result
            episode_return += float(reward)
        total_return += episode_return
    policy.actor.train()
    return total_return / episodes

def train_agent(dataset_name, max_steps, seed, eval_freq=1000):
    np.random.seed(seed); random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    ds = minari.load_dataset(dataset_name, download=True)
    env = ds.recover_environment()
    dset = dataset_to_d4rl(ds)
    state_dim, action_dim = dset["observations"].shape[1], dset["actions"].shape[1]
    max_action = float(env.action_space.high[0])
    stats = compute_stats_from_d4rl(dset)
    cfg = choose_profile(dataset_name, stats, max_action)
    print(f"  Auto-profile loaded: {cfg['label']}")
    if cfg.get("episode_filter_frac", 0.0) > 0:
        dset, _, _ = filter_topk_by_return(ds, keep_frac=cfg["episode_filter_frac"])
    rb = ReplayBuffer(state_dim, action_dim)
    rb.load_from_d4rl(dset)
    mean, std = rb.normalize_states()
    td3_bc_kwargs = {
        "discount", "tau", "policy_noise", "noise_clip", "policy_freq",
        "alpha", "bc_weight", "bc_warmup", "lambda_clip"
    }
    agent_config = {key: cfg[key] for key in td3_bc_kwargs if key in cfg}
    agent = TD3_BC(state_dim, action_dim, max_action, **agent_config)
    eval_scores, eval_steps = [], []
    pbar = tqdm(range(1, max_steps + 1), desc=f"  Seed {seed}", leave=False, ncols=100)
    for t in pbar:
        agent.train_step(rb, batch_size=cfg["batch_size"])
        if t % eval_freq == 0:
            score = eval_policy(agent, env, mean.squeeze(0), std.squeeze(0), seed=seed, episodes=10)
            pbar.set_postfix({"eval_return": f"{score:.1f}"})
            eval_scores.append(score)
            eval_steps.append(t)
    return eval_scores, eval_steps



if __name__ == "__main__":
    NUM_RUNS = 10
    MAX_TRAINING_STEPS = 50_000
    EVAL_FREQ = 1000
    DATASET_NAMES  = [
    
    "mujoco/reacher/medium-v0",
    "mujoco/reacher/expert-v0",
    "mujoco/halfcheetah/expert-v0",
    "mujoco/halfcheetah/medium-v0",
    "mujoco/inverteddoublependulum/expert-v0",
    "mujoco/inverteddoublependulum/medium-v0",
    "mujoco/invertedpendulum/expert-v0",
    "mujoco/invertedpendulum/medium-v0"
 
    ]
    all_results = {}
    for dataset in DATASET_NAMES:
        print(f"\n{'='*25} Running Experiment on: {dataset} {'='*25}")
        run_scores_history, run_steps = [], None
        for i in range(NUM_RUNS):
            current_seed = BASE_SEED + i
            print(f"--- Starting Run {i+1}/{NUM_RUNS} (Seed: {current_seed}) ---")
            scores, steps = train_agent(dataset, MAX_TRAINING_STEPS, current_seed, EVAL_FREQ)
            run_scores_history.append(scores)
            if run_steps is None: run_steps = steps
        scores_np = np.array(run_scores_history)
        all_results[dataset] = {
            "steps": run_steps, "scores_all_runs": scores_np,
            "mean": np.mean(scores_np, axis=0), "std": np.std(scores_np, axis=0)
        }
    print("\n" + "="*30 + " Plotting Results " + "="*30)
    for dataset, results in all_results.items():
        plt.figure(figsize=(12, 7))
        steps, mean, std = results['steps'], results['mean'], results['std']
        plt.plot(steps, mean, label='Mean Return', color='b', linewidth=2)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2, color='b', label='Standard Deviation')
        plt.xlabel("Training Steps", fontsize=14)
        plt.ylabel("Evaluation Return", fontsize=14)
        plt.title(f"TD3+BC Performance on {dataset}\n(Mean & Std. Dev. over {NUM_RUNS} runs)", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
    print("\n" + "="*25 + " Final Performance Summary " + "="*25)
    header = f"{'Dataset':<30} | Final Mean Return (Erwartungswert ± Steuerbreite)"
    print(header); print("-" * len(header))
    for dataset, results in all_results.items():
        final_scores = results['scores_all_runs'][:, -1]
        final_mean, final_std = np.mean(final_scores), np.std(final_scores)
        summary_str = f"{final_mean:.2f} ± {final_std:.2f}"
        print(f"{dataset:<30} | {summary_str}")
