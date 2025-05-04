#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare 3 runs: DQN-PIRL / TD3-PIRL / TD3-Lagrangian-PIRL
"""

import os, glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.plugin_event_accumulator import (
    EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE)


data_root_dir = os.path.expanduser('~/pirl_carla_la_g/logs/MapC')
labels        = ['DQN-PIRL', 'TD3-PIRL', 'TD3-Lagrangian-PIRL']

window_size   = 2000     
min_periods   = 100


data_dirs   = sorted(
    d for d in (os.path.join(data_root_dir, x) for x in os.listdir(data_root_dir))
    if os.path.isdir(d))
event_files = [glob.glob(os.path.join(d, 'events*'))[0] for d in data_dirs]

assert len(event_files) >= len(labels), \
    f'only {len(event_files)} ,but labels  {len(labels)} '


smoothed_rw, smoothed_q0 = [], []

for run_id, (file_path, tag) in enumerate(zip(event_files, labels)):
    print(f'[{run_id}] loading {file_path}')
    acc = EventAccumulator(file_path, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    acc.Reload()

    ep_q0 = pd.Series([t.tensor_proto.float_val[0] for t in acc.Tensors('Episode Q0')])
    ep_rw = pd.Series([t.tensor_proto.float_val[0] for t in acc.Tensors('Episode Reward')])

    smoothed_q0.append(ep_q0.rolling(window_size, min_periods=min_periods).mean())
    smoothed_rw.append(ep_rw.rolling(window_size, min_periods=min_periods).mean())

def plot_metric(series_list, ylabel, ymin, ymax, fname, title,
                win_std=500, min_periods_std=100):
    """
    series_list : List[pd.Series]  
    win_std     : int              
    """

    plt.figure(figsize=(10, 6))
    colors = ['tab:blue', 'tab:red', 'tab:orange']   

    for s, lb, c in zip(series_list, labels, colors):
        plt.plot(s, lw=1.5, color=c, label=lb, zorder=3)

        std = 3 * s.rolling(win_std, min_periods=min_periods_std).std()

        plt.fill_between(
            s.index,
            s - std,
            s + std,
            color=c,
            alpha=0.20,     
            linewidth=0,
            zorder=2       
        )

    plt.xlim([0, 30_000])
    plt.ylim([ymin, ymax])
    plt.xlabel('Episodes')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
plot_metric(smoothed_rw,
            ylabel='Average Episode Reward',
            ymin=-0.05, ymax=1.05,
            fname='Episode_Reward_comparison.png',
            title='PIRL variants – Episode Reward')
