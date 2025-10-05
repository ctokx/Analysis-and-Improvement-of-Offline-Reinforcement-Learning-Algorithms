import random, numpy as np, torch
from tqdm import tqdm
import minari
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
LOG_ROOT = os.path.join(os.path.dirname(__file__), "d3rlpy_logs")

def _list_run_dirs():
    try:
        return set([d for d in os.listdir(LOG_ROOT) if os.path.isdir(os.path.join(LOG_ROOT, d))])
    except FileNotFoundError:
        return set()

def _delete_models_in_dirs(dirs):
    for d in dirs:
        p = os.path.join(LOG_ROOT, d)
        try:
            for f in os.listdir(p):
                if f.startswith("model_") and f.endswith(".d3"):
                    try:
                        os.remove(os.path.join(p, f))
                    except Exception:
                        pass
        except Exception:
            pass
import re

def _step_of(fname: str) -> int:
    m = re.search(r"model_(\d+)\.d3$", fname)
    return int(m.group(1)) if m else -1
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# d3rlpy imports
from d3rlpy.algos import CQL, IQL, CQLConfig, IQLConfig
from d3rlpy.dataset import MDPDataset
 
try:
    from d3rlpy import load_learnable as d3_load
except Exception:
    d3_load = None


# --- Repro & device ---
BASE_SEED = 42
np.random.seed(BASE_SEED); random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed(BASE_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================================================================
# --- DATA HELPERS (mirrors rl/bc.py) ---
# ==============================================================================

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


# ==============================================================================
# --- ALGO PROFILES (CQL / IQL) ---
# ==============================================================================

def choose_profile_common(dataset_name: str):
    n = dataset_name.lower()
    cfg = dict(
        label="DEFAULT",
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        episode_filter_frac=0.0,
        target=900.0,
    )
    if "expert" in n:
        cfg.update(dict(label="EXPERT", batch_size=256, target=1000.0))
    elif "medium" in n:
        cfg.update(dict(label="MEDIUM", batch_size=1024, episode_filter_frac=0.70, target=1000.0))
    else:
        cfg.update(dict(label="MEDIUMLIKE", batch_size=1024))
    return cfg


def choose_profile_cql(dataset_name: str):
    cfg = choose_profile_common(dataset_name)
    # CQL-specific knobs (continuous control defaults)
    cfg.update(dict(
        alpha=1.0,  # conservative regularizer weight
        n_action_samples=10,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        soft_q_backup=True,
    ))
    return cfg


def choose_profile_iql(dataset_name: str):
    cfg = choose_profile_common(dataset_name)
    # IQL-specific knobs (per original paper defaults)
    cfg.update(dict(
        expectile=0.7,
        temperature=3.0,
        awr_beta=3.0,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        value_learning_rate=3e-4,
    ))
    return cfg


# ==============================================================================
# --- EVALUATION (rollout env) ---
# ==============================================================================

def eval_policy_d3rlpy(algo, env, seed, episodes=10):
    total_return = 0.0
    for i in range(episodes):
        try:
            state, _ = env.reset(seed=seed + 100 + i)
        except TypeError:
            random.seed(seed + 100 + i); np.random.seed(seed + 100 + i)
            state = env.reset()
        done, episode_return = False, 0.0
        while not done:
            # d3rlpy expects batched input; add batch dim and take first action
            action = algo.predict(np.asarray(state, dtype=np.float32)[None, :])[0]
            step_result = env.step(action)
            if len(step_result) == 5:
                state, reward, term, trunc, _ = step_result
                done = term or trunc
            else:
                state, reward, done, _ = step_result
            episode_return += float(reward)
        total_return += episode_return
    return total_return / episodes


# ==============================================================================
# --- TRAINERS (CQL / IQL) ---
# ==============================================================================

def build_mdp_dataset_from_d4rl(d4rl_like):
    obs = d4rl_like["observations"].astype(np.float32)
    act = d4rl_like["actions"].astype(np.float32)
    rew = d4rl_like["rewards"].astype(np.float32)
    term = (d4rl_like["terminals"].astype(np.float32) > 0.5).astype(np.bool_)
    tout = (d4rl_like["timeouts"].astype(np.float32) > 0.5).astype(np.bool_)
    # Some d3rlpy versions don't support episode_terminals; fold timeouts into terminals
    terminals_all = np.logical_or(term, tout)
    return MDPDataset(
        observations=obs,
        actions=act,
        rewards=rew,
        terminals=terminals_all,
    )


def train_agent_cql(dataset_name, max_steps, seed, eval_freq=1000, save_checkpoints=False, gpu_id=None):
    np.random.seed(seed); random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    # Pin process to a specific GPU if requested
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass

    ds = minari.load_dataset(dataset_name, download=True)
    env = ds.recover_environment()
    dset = dataset_to_d4rl(ds)

    cfg = choose_profile_cql(dataset_name)
    print(f"  Auto-profile loaded (CQL): {cfg['label']}")
    if cfg.get("episode_filter_frac", 0.0) > 0:
        dset, _, _ = filter_topk_by_return(ds, keep_frac=cfg["episode_filter_frac"])

    mdp = build_mdp_dataset_from_d4rl(dset)

    cql_config = CQLConfig()
    dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    algo = CQL(config=cql_config, device=dev_str, enable_ddp=False)

    # ensure single run directory for our own checkpoints
    if save_checkpoints:
        run_dir = os.path.join(os.path.dirname(__file__), "d3rlpy_logs", f"CQL_custom_{dataset_name.replace('/', '_')}_seed{seed}")
        os.makedirs(run_dir, exist_ok=True)

    eval_scores, eval_steps = [], []
    # Single continuous training; d3rlpy will create one run dir with model_<step>.d3 per epoch
    before_dirs = _list_run_dirs()
    algo.fit(mdp, n_steps=max_steps, n_steps_per_epoch=eval_freq)
    after_dirs = _list_run_dirs()
    new_dirs = sorted(list(after_dirs - before_dirs))
    run_dir_created = os.path.join(LOG_ROOT, new_dirs[-1]) if new_dirs else None
    # Evaluate each checkpoint file in that run dir
    if run_dir_created and d3_load is not None:
        ckpts = [f for f in os.listdir(run_dir_created) if f.startswith("model_") and f.endswith(".d3")]
        ckpts.sort(key=_step_of)
        for ck in ckpts:
            loaded = d3_load(os.path.join(run_dir_created, ck))
            score = eval_policy_d3rlpy(loaded, env, seed=seed, episodes=10)
            eval_steps.append(_step_of(ck))
            eval_scores.append(score)
    return eval_scores, eval_steps


def train_agent_iql(dataset_name, max_steps, seed, eval_freq=1000, save_checkpoints=False, gpu_id=None):
    np.random.seed(seed); random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass

    ds = minari.load_dataset(dataset_name, download=True)
    env = ds.recover_environment()
    dset = dataset_to_d4rl(ds)

    cfg = choose_profile_iql(dataset_name)
    print(f"  Auto-profile loaded (IQL): {cfg['label']}")
    if cfg.get("episode_filter_frac", 0.0) > 0:
        dset, _, _ = filter_topk_by_return(ds, keep_frac=cfg["episode_filter_frac"])

    mdp = build_mdp_dataset_from_d4rl(dset)

    iql_config = IQLConfig()
    dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    algo = IQL(config=iql_config, device=dev_str, enable_ddp=False)

    if save_checkpoints:
        run_dir = os.path.join(os.path.dirname(__file__), "d3rlpy_logs", f"IQL_custom_{dataset_name.replace('/', '_')}_seed{seed}")
        os.makedirs(run_dir, exist_ok=True)

    eval_scores, eval_steps = [], []
    before_dirs = _list_run_dirs()
    algo.fit(mdp, n_steps=max_steps, n_steps_per_epoch=eval_freq)
    after_dirs = _list_run_dirs()
    new_dirs = sorted(list(after_dirs - before_dirs))
    run_dir_created = os.path.join(LOG_ROOT, new_dirs[-1]) if new_dirs else None
    if run_dir_created and d3_load is not None:
        ckpts = [f for f in os.listdir(run_dir_created) if f.startswith("model_") and f.endswith(".d3")]
        ckpts.sort(key=_step_of)
        for ck in ckpts:
            loaded = d3_load(os.path.join(run_dir_created, ck))
            score = eval_policy_d3rlpy(loaded, env, seed=seed, episodes=10)
            eval_steps.append(_step_of(ck))
            eval_scores.append(score)
    return eval_scores, eval_steps


# ==============================================================================
# --- MAIN EXPERIMENT RUNNER ---
# ==============================================================================

if __name__ == "__main__":
    NUM_RUNS = 10
    MAX_TRAINING_STEPS = 50_000
    EVAL_FREQ = 1000
    DATASET_NAMES = [
        "mujoco/walker2d/medium-v0",
        "mujoco/walker2d/expert-v0",
        "mujoco/swimmer/medium-v0",
        "mujoco/swimmer/expert-v0",
        "mujoco/reacher/medium-v0",
        "mujoco/reacher/expert-v0",
        "mujoco/pusher/medium-v0",
        "mujoco/pusher/expert-v0",
        "mujoco/invertedpendulum/expert-v0",
        "mujoco/invertedpendulum/medium-v0",
        "mujoco/inverteddoublependulum/expert-v0",
        "mujoco/inverteddoublependulum/medium-v0",
        "mujoco/hopper/medium-v0",
        "mujoco/hopper/expert-v0",
        "mujoco/halfcheetah/medium-v0",
        # "mujoco/halfcheetah/expert-v0",

    ]

    # GPU-aware parallelism: read GPU_IDS env (e.g., "0,1,2,3") or detect all
    gpu_ids_env = os.environ.get("GPU_IDS", "").strip()
    if gpu_ids_env:
        gpu_ids = [int(x) for x in gpu_ids_env.split(',') if x.strip() != ""]
    else:
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
    PROCS_PER_GPU = int(os.environ.get("PROCS_PER_GPU", "1"))
    PROCS_PER_GPU = max(1, PROCS_PER_GPU)
    MAX_WORKERS = min(NUM_RUNS, max(1, len(gpu_ids) * PROCS_PER_GPU))

    for algo_name, trainer in [("CQL", train_agent_cql), ("IQL", train_agent_iql)]:
        all_results = {}
        for dataset in DATASET_NAMES:
            print(f"\n{'='*25} Running {algo_name} on: {dataset} {'='*25}")
            run_scores_history, run_steps = [], None
            futures = []
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
                for i in range(NUM_RUNS):
                    current_seed = BASE_SEED + i
                    if gpu_ids == [None]:
                        assigned_gpu = None
                    else:
                        assigned_gpu = gpu_ids[(i // PROCS_PER_GPU) % len(gpu_ids)]
                    futures.append(ex.submit(trainer, dataset, MAX_TRAINING_STEPS, current_seed, EVAL_FREQ, False, assigned_gpu))
                for idx, fut in enumerate(as_completed(futures), 1):
                    try:
                        scores, steps = fut.result()
                        run_scores_history.append((scores, steps))
                        if run_steps is None:
                            run_steps = steps
                        print(f"  Completed run {idx}/{NUM_RUNS}")
                    except Exception as e:
                        print(f"  Run failed: {e}")
            if not run_scores_history:
                continue
            # Aggregate per step across possibly different-length runs
            step_to_scores = {}
            all_steps = set()
            for scores, steps in run_scores_history:
                all_steps.update(steps)
            all_steps = sorted(all_steps)
            for s in all_steps:
                vals = []
                for scores, steps in run_scores_history:
                    try:
                        idx_s = steps.index(s)
                        vals.append(scores[idx_s])
                    except ValueError:
                        continue
                step_to_scores[s] = vals
            mean = [float(np.mean(step_to_scores[s])) if step_to_scores[s] else float('nan') for s in all_steps]
            std = [float(np.std(step_to_scores[s])) if step_to_scores[s] else float('nan') for s in all_steps]
            all_results[dataset] = {
                "steps": all_steps,
                "mean": np.array(mean),
                "std": np.array(std),
                "step_to_scores": step_to_scores,
            }

        print("\n" + "="*30 + f" Plotting Results ({algo_name}) " + "="*30)
        for dataset, results in all_results.items():
            plt.figure(figsize=(12, 7))
            steps, mean, std = results['steps'], results['mean'], results['std']
            plt.plot(steps, mean, label='Mean Return', color='b', linewidth=2)
            plt.fill_between(steps, mean - std, mean + std, alpha=0.2, color='b', label='Standard Deviation')
            plt.xlabel("Training Steps", fontsize=14)
            plt.ylabel("Evaluation Return", fontsize=14)
            plt.title(f"{algo_name} Performance on {dataset}\n(Mean & Std. Dev. over {NUM_RUNS} runs)", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            out_dir = os.path.join(os.path.dirname(__file__), "plots", algo_name)
            os.makedirs(out_dir, exist_ok=True)
            filename = f"{dataset.replace('/', '_')}.png"
            plt.savefig(os.path.join(out_dir, filename), dpi=150)
            plt.close()

        print("\n" + "="*25 + f" Final Performance Summary ({algo_name}) " + "="*25)
        header = f"{'Dataset':<30} | Final Mean Return (mean ± std)"
        print(header); print("-" * len(header))
        for dataset, results in all_results.items():
            steps = results['steps']
            step_to_scores = results['step_to_scores']
            last_step = steps[-1]
            final_list = step_to_scores.get(last_step, [])
            final_mean = float(np.mean(final_list)) if final_list else float('nan')
            final_std = float(np.std(final_list)) if final_list else float('nan')
            summary_str = f"{final_mean:.2f} ± {final_std:.2f}"
            print(f"{dataset:<30} | {summary_str}")


