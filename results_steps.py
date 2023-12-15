import torch
import numpy as np
import random
import pickle
import pandas as pd
from pathlib import Path
from arguments import set_args
import matplotlib.pyplot as plt

dataset = "meta_dataset"# args["dataset"] #rmnist, cifar10, femnist, meta_dataset, complete_meta_dataset
partition = "None" #None, shards, dirichlet

folders = [f"_saved_models/{dataset}/{partition}_steps_new/seed0/",
           f"_saved_models/{dataset}/{partition}_steps_new/seed1/", #seed5 for dirichlet
           f"_saved_models/{dataset}/{partition}_steps_new/seed2/",
           f"_saved_models/{dataset}/{partition}_steps_new/seed5/",
           f"_saved_models/{dataset}/{partition}_steps_new/seed6/",
           ]
PATH = Path(f"_saved_models/{dataset}/{partition}_steps_new")
PATH.mkdir(parents=True, exist_ok=True)
print(PATH)
methods = ["perfedavg", "ditto", "fedrep", "proposed_c1"]

# Load accuracies
dict_accuracies_mean, dict_accuracies_std = {}, {}
for i, name in enumerate(methods):
    avg_accs = []
    for f in folders:
        accs = []
        for s in range(0, 200, 10):
            if s==0: file_dir = f + name + f'/step{s+1}.pt'
            else: file_dir = f + name + f'/step{s}.pt'
            checkpoint = torch.load(file_dir)
            accs.append(checkpoint['test_acc'])
        avg_accs.append(accs)
    dict_accuracies_mean[name] = np.mean(avg_accs, axis=0)
    dict_accuracies_std[name] = np.std(avg_accs, axis=0)

x = list(range(0, 200, 10))
plt.plot(x, dict_accuracies_mean["perfedavg"], marker='o', color="tab:blue", label="Per-FedAvg")
#plt.fill_between(x, dict_accuracies_mean["perfedavg"]-dict_accuracies_std["perfedavg"], dict_accuracies_mean["perfedavg"]+dict_accuracies_std["perfedavg"], color='tab:blue', alpha=0.2)
plt.plot(x, dict_accuracies_mean["ditto"], marker='o', color = "tab:orange", label="Ditto")
#plt.fill_between(x, dict_accuracies_mean["ditto"]-dict_accuracies_std["ditto"], dict_accuracies_mean["ditto"]+dict_accuracies_std["ditto"], color='tab:orange', alpha=0.2)
plt.plot(x, dict_accuracies_mean["fedrep"], marker='o', color = "tab:green", label="FedRep")
#plt.fill_between(x, dict_accuracies_mean["fedrep"]-dict_accuracies_std["fedrep"], dict_accuracies_mean["fedrep"]+dict_accuracies_std["fedrep"], color='tab:green', alpha=0.2)
plt.plot(x, dict_accuracies_mean["proposed_c1"], marker='o', color="tab:red", label="CAFeMe")
#plt.fill_between(x, dict_accuracies_mean["proposed_c1"]-dict_accuracies_std["proposed_c1"], dict_accuracies_mean["proposed_c1"]+dict_accuracies_std["proposed_c1"], color='tab:red', alpha=0.2)
plt.ylabel("Average accuracy")
plt.xlabel("Global steps")
plt.legend()
plt.savefig(PATH / 'avg_comparison_steps.png')
plt.show()
