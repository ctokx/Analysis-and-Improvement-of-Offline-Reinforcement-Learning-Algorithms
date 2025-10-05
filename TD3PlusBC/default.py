
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm

import d3rlpy
import minari
import gymnasium as gym
from d3rlpy.algos import TD3PlusBCConfig
from d3rlpy.metrics import EnvironmentEvaluator



def run_single_modern_experiment(dataset_name, seed, training_steps, eval_interval, device):
    """Runs a single d3rlpy experiment using the modern Minari/Gymnasium stack."""
    print(f"    Run Seed {seed}:")
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    minari_dataset = minari.load_dataset(dataset_name, download=True)
    eval_env = minari_dataset.recover_environment()

    observations, actions, rewards, terminals, timeouts = [], [], [], [], []
    for episode in minari_dataset.iterate_episodes():
        observations.append(episode.observations)
        actions.append(episode.actions)
        rewards.append(episode.rewards)
        terminals.append(episode.terminations)
        timeouts.append(episode.truncations)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        rewards=np.concatenate(rewards),
        terminals=np.concatenate(terminals),
        timeouts=np.concatenate(timeouts)
    )

    config = TD3PlusBCConfig(
        batch_size=256, alpha=2.5, actor_learning_rate=3e-4,
        critic_learning_rate=3e-4, tau=0.005, gamma=0.99,
    )
    algo = config.create(device=device)
    evaluator = EnvironmentEvaluator(eval_env, n_trials=10)
    experiment_name = f"{dataset_name.replace('/', '_')}_seed_{seed}"
    
    history = algo.fit(
        dataset,
        n_steps=training_steps,
        n_steps_per_epoch=eval_interval,
        evaluators={"environment": evaluator},
        experiment_name=experiment_name,
        show_progress=False,
        save_interval=training_steps + 1
    )

    steps, returns = [], []
    for epoch, metrics in history:
        if 'environment' in metrics:
            current_step = epoch * eval_interval
            steps.append(current_step)
            returns.append(metrics['environment'])

    if steps:
        performance_df = pd.DataFrame({"step": steps, "return": returns})
        return {"performance": performance_df}
    
    return {"performance": None}

def plot_analysis(dataset_name, all_run_logs, num_runs):
    """Robust plotting function to prevent blank charts."""
    short_name = "/".join(dataset_name.split('/')[1:]) # Format name for title
    plt.figure(figsize=(12, 7))
    perf_logs = [log['performance'] for log in all_run_logs if log.get('performance') is not None and not log['performance'].empty]
    
    if perf_logs:
        df_perf = pd.concat(perf_logs).groupby('step').agg(
            mean_return=('return', 'mean'), std_return=('return', 'std')
        ).reset_index()

        if not df_perf.empty:
            plt.plot(df_perf['step'], df_perf['mean_return'], label='Mean Return', color='b', linewidth=2)
            plt.fill_between(
                df_perf['step'], df_perf['mean_return'] - df_perf['std_return'],
                df_perf['mean_return'] + df_perf['std_return'],
                alpha=0.2, color='b', label='Standard Deviation'
            )
            plt.legend(fontsize=12)
        else: print("Warning: Performance data was empty after processing. Skipping plot.")
    else: print("Warning: No valid performance logs found to plot.")

    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Evaluation Return", fontsize=14)
    plt.title(f"TD3+BC on {short_name}\n(Mean & Std. Dev. over {num_runs} runs)", fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


NUM_RUNS = 10
TRAINING_STEPS = 50_000
EVAL_INTERVAL = 1000


DATASETS = [
    "mujoco/reacher/medium-v0",
    "mujoco/reacher/expert-v0",
    "mujoco/invertedpendulum/medium-v0",
    "mujoco/invertedpendulum/expert-v0",
    "mujoco/inverteddoublependulum/medium-v0",
    "mujoco/inverteddoublependulum/expert-v0",
    "mujoco/hopper/medium-v0",
    "mujoco/hopper/expert-v0"
]


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {device.upper()} ---")

all_results_summary = {}
if os.path.exists('d3rlpy_logs'):
    shutil.rmtree('d3rlpy_logs')

for dataset_name in DATASETS:
    short_name = "/".join(dataset_name.split('/')[1:])
    print(f"\n{'='*30} Running Benchmark on: {dataset_name} {'='*30}")
    all_run_logs = []
    final_scores_for_summary = []
    
    with tqdm(total=NUM_RUNS, desc=f"Training on {short_name}") as pbar:
        for i in range(NUM_RUNS):
            current_seed = 42 + i
            run_logs = run_single_modern_experiment(dataset_name, current_seed, TRAINING_STEPS, EVAL_INTERVAL, device)
            all_run_logs.append(run_logs)
            if run_logs.get('performance') is not None and not run_logs['performance'].empty:
                final_scores_for_summary.append(run_logs['performance']['return'].iloc[-1])
            pbar.update(1)

    plot_analysis(dataset_name, all_run_logs, NUM_RUNS)
    
    if final_scores_for_summary:
        mean_score = np.mean(final_scores_for_summary)
        std_score = np.std(final_scores_for_summary)
        all_results_summary[dataset_name] = {"mean": mean_score, "std": std_score}
        
        print("\n" + "-"*25 + " Summary for this Dataset " + "-"*25)
        summary_str = f"{mean_score:.2f} ± {std_score:.2f}"
        print(f"{short_name:<30} | {summary_str}")
        print("-" * (25 * 2 + 28))


print("\n" + "="*25 + " Final Overall Performance Summary " + "="*25)
header = f"{'Dataset':<30} | Final Mean Return (Mean ± Std)"
print(header); print("-" * len(header))
for dataset, results in all_results_summary.items():
    short_name = "/".join(dataset.split('/')[1:])
    summary_str = f"{results['mean']:.2f} ± {results['std']:.2f}"
    print(f"{short_name:<30} | {summary_str}")
