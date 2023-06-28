import torch
import numpy as np
import random
import pickle
import pandas as pd
from pathlib import Path
from arguments import set_args
import matplotlib.pyplot as plt

dataset = "cifar10"# args["dataset"] #rmnist, cifar10, femnist, meta_dataset, complete_meta_dataset
partition = "shards" #None, shards, dirichlet

folders = [f"_saved_models/{dataset}/{partition}_steps/seed0/",
           f"_saved_models/{dataset}/{partition}_steps/seed1/", #seed5 for dirichlet
           f"_saved_models/{dataset}/{partition}_steps/seed2/",
           f"_saved_models/{dataset}/{partition}_steps/seed3/",
           f"_saved_models/{dataset}/{partition}_steps/seed4/",
           ]
PATH = Path(f"_saved_models/{dataset}/{partition}_steps")
PATH.mkdir(parents=True, exist_ok=True)
print(PATH)
methods = ["fedavg", "fedavg_ft", "perfedavg", "ifca", "ifca_ft", "proposed_c1"]

# Load accuracies
dict_accuracies_mean, dict_accuracies_std = {}, {}
for i, name in enumerate(methods):
    avg_accs = []
    for f in folders:
        accs = []
        for s in range(100, 1000+1, 100):
            file_dir = f + name + f'/step{s}.pt'
            checkpoint = torch.load(file_dir)
            accs.append(checkpoint['test_acc'])
        avg_accs.append(accs)
    dict_accuracies_mean[name] = np.mean(avg_accs, axis=0)
    dict_accuracies_std[name] = np.std(avg_accs, axis=0)

x = list(range(100, 1000+1, 100))
plt.plot(x, dict_accuracies_mean["perfedavg"], marker='o', label="Per-FedAvg")
plt.plot(x, dict_accuracies_mean["proposed_c1"], marker='o', label="CAFeMe")
plt.ylabel("Average accuracy")
plt.xlabel("Global steps")
plt.legend()
plt.show()
