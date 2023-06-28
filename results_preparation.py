import torch
import numpy as np
import random
import pickle
import pandas as pd
from pathlib import Path
from arguments import set_args
import matplotlib.pyplot as plt

dataset = "cifar10"# args["dataset"] #rmnist, cifar10, femnist, meta_dataset, complete_meta_dataset
partition = "dirichlet"

folders = [f"_saved_models/{dataset}/{partition}/dirichlet_0.3/seed0/",
           f"_saved_models/{dataset}/{partition}/dirichlet_0.3/seed5/", #seed5 for dirichlet
           f"_saved_models/{dataset}/{partition}/dirichlet_0.3/seed2/",
           f"_saved_models/{dataset}/{partition}/dirichlet_0.3/seed3/",
           f"_saved_models/{dataset}/{partition}/dirichlet_0.3/seed4/",
           ]
PATH = Path(f"_saved_models/{dataset}/{partition}/dirichlet_0.3")
PATH.mkdir(parents=True, exist_ok=True)
print(PATH)
methods = ["fedavg", "fedavg_ft", "perfedavg", "ifca", "ifca_ft", "ifca_sharing", "ditto", "proposed_c1"]

# Load accuracies
dict_accuracies_mean, dict_accuracies_std = {}, {}
for i, name in enumerate(methods):
    avg_accuracies = []
    for f in folders:
        file_dir = f + f"acc_{name}.pkl"
        file = open(file_dir, 'rb')
        data_loaded = pickle.load(file)
        file.close()
        if name == "proposed_c1" or name == "perfedavg" or name == "ditto": print(f"{name}: {data_loaded[-1]}")
        avg_accuracies.append(data_loaded)
    dict_accuracies_mean[name] = np.mean(avg_accuracies, axis=0)
    dict_accuracies_std[name] = np.std(avg_accuracies, axis=0)


# Create table
df = pd.DataFrame(columns=["mean", "std"], index=methods)
for name in methods:
    mean_acc = dict_accuracies_mean[name]
    std_acc = dict_accuracies_std[name]
    try:
        if len(mean_acc) > 3: # pfl method
            mean_acc = np.mean(mean_acc[-3:])
            std_acc = np.mean(std_acc[-3:])
    except:
        pass
    df.loc[name] = [round(100 * np.mean(mean_acc), 2), round(100 * np.mean(std_acc), 2)]
df.to_csv(PATH / f"avg_comparison.csv", encoding='utf-8')

# Plot personalization accuracies
plt.plot(dict_accuracies_mean["fedavg_ft"], label="fedavg_ft", c='tab:blue')
plt.fill_between(range(51), dict_accuracies_mean["fedavg_ft"]-dict_accuracies_std["fedavg_ft"], dict_accuracies_mean["fedavg_ft"]+dict_accuracies_std["fedavg_ft"], color='tab:blue', alpha=0.2)
plt.plot(dict_accuracies_mean["perfedavg"], label="perfedavg", c="tab:orange")
plt.fill_between(range(51), dict_accuracies_mean["perfedavg"]-dict_accuracies_std["perfedavg"], dict_accuracies_mean["perfedavg"]+dict_accuracies_std["perfedavg"], color='tab:orange', alpha=0.2)
plt.plot(dict_accuracies_mean["ifca_ft"], label="ifca_ft", c="tab:green")
plt.fill_between(range(51), dict_accuracies_mean["ifca_ft"]-dict_accuracies_std["ifca_ft"], dict_accuracies_mean["ifca_ft"]+dict_accuracies_std["ifca_ft"], color='tab:green', alpha=0.2)
plt.plot(dict_accuracies_mean["ditto"], label="ditto", c="tab:purple")
plt.fill_between(range(51), dict_accuracies_mean["ditto"]-dict_accuracies_std["ditto"], dict_accuracies_mean["ditto"]+dict_accuracies_std["ditto"], color='tab:purple', alpha=0.2)
plt.plot(dict_accuracies_mean["proposed_c1"], label="proposed", c="tab:red")
plt.fill_between(range(51), dict_accuracies_mean["proposed_c1"]-dict_accuracies_std["proposed_c1"], dict_accuracies_mean["proposed_c1"]+dict_accuracies_std["proposed_c1"], color='tab:red', alpha=0.2)
plt.ylabel("Average accuracy")
plt.xlabel("Personalization steps")
plt.legend()
plt.savefig(PATH / 'avg_comparison_pfl.png')
plt.show()
