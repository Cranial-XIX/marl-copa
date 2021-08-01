import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import sys

from scipy.signal import savgol_filter

lw = 2
fs = 20

colormap = [
    "tab:blue",
    "tab:brown",
    "tab:pink",
    "tab:orange",
    "tab:green",
    "yellow",
    "tab:red",
    "deeppink",
    "darkcyan",
    "slateblue",
    "tab:gray",
    "royalblue",
]
markermap = ["x", "s", "v", "o", "d", "x", "*"]

marker_args = dict(
        markerfacecolor = "None",
        markeredgewidth=5,
        ms=14)

experiments = [
    "aqmix+ctr4",
    "aiqmix+ctr4",
    "aqmix+coach+ctr4",
    "aqmix+full+ctr4",
    "aqmix+period+ctr4",
    "aqmix+coach+vi+ctr2+vi0.001",
    "aqmix+coach+vi+ctr4+vi0.001",
    "aqmix+coach+vi+ctr8+vi0.001",
]
labels = ["aqmix", "aiqmix", "aqmix+coach", "aqmix+full", "aqmix+period", "copa(T=2)", "copa(T=4)", "copa(T=8)"]

runs = [0,1,2]
c = 0
color_dict = {}
for exp,lb in zip(experiments, labels):
    color_dict[exp] = c
    c += 1

def smooth(y, n=100):
    y = np.array(y)
    y_ = savgol_filter(y, 101, 3)
    return y_

fig = plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
ax1 = plt.subplot(111)
xx = None
env_id = "mpe"

for i, (exp, lb) in enumerate(zip(experiments, labels)):
    x = None; Y = []
    min_len = 1000000000
    for run in runs:
        filename = f"results/{env_id}/{exp}/run{run}/logs/stats.npy"
        if not os.path.exists(filename):
            continue
        try:
            data = np.load(filename, allow_pickle=True).item()
            x, y = zip(*data.get('reward'))
            Y.append(smooth(y))
        except:
            continue
        min_len = min(min_len, len(y))
    if x is None:
        continue
    x = np.array(x[:min_len]) / 1e6
    Y = np.array([y[:min_len] for y in Y])
    mu = Y.mean(0)
    std = Y.std(0)
    ax1.plot(x, mu, color=colormap[color_dict[exp]], alpha=1.0, lw=lw, ls="-", label=labels[i])
    ax1.fill_between(x, mu-std, mu+std, color=colormap[color_dict[exp]], alpha=0.2)

ax1.set_xlabel("Timestep (mil)", fontsize=fs)
ax1.set_ylabel("Reward", fontsize=fs)
ax1.legend()

for label in ax1.get_xticklabels():
    label.set_fontsize(fs)
for label in ax1.get_yticklabels():
    label.set_fontsize(fs)

plot_type = "all"
plt.ylim(-10, 120)
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig(f"training.pdf", format="pdf")
plt.close()
