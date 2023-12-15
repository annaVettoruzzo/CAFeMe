import torch
import numpy as np
import random
import pickle
import pandas as pd
from pathlib import Path
from arguments import set_args
import matplotlib.pyplot as plt

dataset = "femnist"# args["dataset"] #rmnist, cifar10, femnist, meta_dataset, complete_meta_dataset
partition = "None"

folders = [f"_saved_models/{dataset}/{partition}/seed0/",
           #f"_saved_models/{dataset}/{partition}/seed1/", #seed5 for dirichlet
           f"_saved_models/{dataset}/{partition}/seed1/",
           f"_saved_models/{dataset}/{partition}/seed3/",
           f"_saved_models/{dataset}/{partition}/seed4/",
           ]
#PATH = Path(f"_saved_models/{dataset}/{partition}")
#PATH.mkdir(parents=True, exist_ok=True)
#print(PATH)
methods = ["fedavg", "fedavg_ft", "perfedavg", "ifca", "ifca_ft", "ditto", "fedrep", "proposed_c1"]

# Load accuracies
dict_accuracies_mean, dict_accuracies_std = {}, {}
for i, name in enumerate(methods):
    avg_accuracies = []
    for f in folders:
        file_dir = f + f"acc_{name}.pkl"
        file = open(file_dir, 'rb')
        data_loaded = pickle.load(file)
        file.close()
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
#df.to_csv(PATH / f"avg_comparison.csv", encoding='utf-8')

# Plot personalization accuracies
fig = plt.figure(figsize=(10,8))
plt.rcParams.update({'font.size': 26})
plt.plot(dict_accuracies_mean["fedavg_ft"], label="FedAvg_FT", c='tab:pink', linewidth=2)
plt.fill_between(range(51), dict_accuracies_mean["fedavg_ft"]-dict_accuracies_std["fedavg_ft"], dict_accuracies_mean["fedavg_ft"]+dict_accuracies_std["fedavg_ft"], color='tab:pink', alpha=0.2)
plt.plot(dict_accuracies_mean["ifca_ft"], label="IFCA-FT", c="tab:purple", linewidth=2)
plt.fill_between(range(51), dict_accuracies_mean["ifca_ft"]-dict_accuracies_std["ifca_ft"], dict_accuracies_mean["ifca_ft"]+dict_accuracies_std["ifca_ft"], color='tab:purple', alpha=0.2)
plt.plot(dict_accuracies_mean["ditto"], label="Ditto", c="tab:orange", linewidth=2)
plt.fill_between(range(51), dict_accuracies_mean["ditto"]-dict_accuracies_std["ditto"], dict_accuracies_mean["ditto"]+dict_accuracies_std["ditto"], color='tab:orange', alpha=0.2)
plt.plot(dict_accuracies_mean["fedrep"], label="FedRep", c="tab:green", linewidth=2)
plt.fill_between(range(51), dict_accuracies_mean["fedrep"]-dict_accuracies_std["fedrep"], dict_accuracies_mean["fedrep"]+dict_accuracies_std["fedrep"], color='tab:green', alpha=0.2)
plt.plot(dict_accuracies_mean["perfedavg"], label="Per-FedAvg", c="tab:blue", linewidth=2)
plt.fill_between(range(51), dict_accuracies_mean["perfedavg"]-dict_accuracies_std["perfedavg"], dict_accuracies_mean["perfedavg"]+dict_accuracies_std["perfedavg"], color='tab:blue', alpha=0.2)
plt.plot(dict_accuracies_mean["proposed_c1"], label="CAFeMe", c="tab:red", linewidth=2)
plt.fill_between(range(51), dict_accuracies_mean["proposed_c1"]-dict_accuracies_std["proposed_c1"], dict_accuracies_mean["proposed_c1"]+dict_accuracies_std["proposed_c1"], color='tab:red', alpha=0.2)
plt.ylabel("Average accuracy")
plt.xlabel("Personalization steps")
plt.title("FEMNIST")
#plt.legend(loc='lower right')
#plt.savefig(PATH / 'avg_comparison_pfl.png')
plt.show()

"""
################## Different modulations #####################
proposed_methods = ['proposed_c0', 'proposed_c1', 'proposed_c2']
# Load accuracies
dict_accuracies_mean, dict_accuracies_std = {}, {}
for i, name in enumerate(proposed_methods):
    avg_accuracies = []
    for f in folders:
        file_dir = f + f"acc_{name}.pkl"
        file = open(file_dir, 'rb')
        data_loaded = pickle.load(file)
        file.close()
        avg_accuracies.append(data_loaded)
    dict_accuracies_mean[name] = np.mean(avg_accuracies, axis=0)
    dict_accuracies_std[name] = np.std(avg_accuracies, axis=0)

# Create table
df = pd.DataFrame(columns=["mean", "std"], index=proposed_methods)
for name in proposed_methods:
    mean_acc = dict_accuracies_mean[name]
    std_acc = dict_accuracies_std[name]
    try:
        if len(mean_acc) > 3: # pfl method
            mean_acc = np.mean(mean_acc[-3:])
            std_acc = np.mean(std_acc[-3:])
    except:
        pass
    df.loc[name] = [round(100 * np.mean(mean_acc), 2), round(100 * np.mean(std_acc), 2)]
df.to_csv(PATH / f"modulation_comparison.csv", encoding='utf-8')

################## Different data heterogeneity - RMNIST #####################
# Create a figure and three subplots for the histograms
data = [[95.09, 95.14, 94.22, 89.47, 87.08],
        [95.52, 95.06, 93.65, 89.84, 88.52],
        [96.10, 96.20, 95.6, 89.20, 90.44],
        [95.33, 96.30, 95.10, 89.85, 90.95],
        [84.21, 84.45, 81.55, 74.57, 83.86],
        [96.21, 96.10, 95.5, 94.31, 98.82]
]

error = [[0.5, 0.11, 0.71, 1.43, 3.98],
        [0.53, 0.73, 0.75, 1.21, 3.66],
        [0.36, 0.7, 0.5, 1.5, 1.89],
        [0.22, 0.5, 0.23, 1.5, 2.06],
        [1.03, 0.45, 0.21, 3.12, 2.82],
        [0.51, 0.43, 0.24, 1.0, 0.13]
]
X = np.arange(5)

fig = plt.figure(figsize=(12,6))
#ax = fig.add_axes([0,0,1,1])
plt.rcParams.update({'font.size': 20})
plt.bar(X + 0.00, data[0], yerr=error[0], align='center', color="tab:pink", alpha=0.9, width = 0.1, ecolor="tab:pink", capsize=2)
plt.bar(X + 0.1, data[1], yerr=error[1], align='center', color="tab:purple", alpha=0.9, width = 0.1, ecolor="tab:purple", capsize=2)
plt.bar(X + 0.2, data[2], yerr=error[2], align='center', color='tab:orange', alpha=0.9, width = 0.1, ecolor="tab:orange", capsize=2)
plt.bar(X + 0.3, data[3], yerr=error[3], align='center', color="tab:green", alpha=0.9, width = 0.1, ecolor="tab:green", capsize=2)
plt.bar(X + 0.4, data[4], yerr=error[4], align='center', color="tab:blue", alpha=0.9, width = 0.1, ecolor="tab:blue", capsize=2)
plt.bar(X + 0.5, data[5], yerr=error[5], align='center', color="tab:red", alpha=0.9, width = 0.1, ecolor="tab:red", capsize=2)
plt.ylim([70, 100])
plt.ylabel('Average accuracy')
plt.xlabel('Partition schemes')
plt.xticks([r + 0.25 for r in range(5)],
        ['None', 'Dir α = 1000', 'Dir α = 5', 'Dir α = 0.3', 'Shards'])
plt.legend(labels=['FedAvg-FT', 'IFCA-FT', 'Ditto', 'FedRep', 'Per-FedAvg', 'CAFeMe'], loc='center left', bbox_to_anchor=(1, 0.5))
plt.vlines(x=3.75, ymin=0, ymax=99, colors='k', ls='--', alpha = 0.5)
plt.title("RMNIST")
plt.tight_layout()
#plt.savefig(PATH / 'data_heterogenity.png')
plt.show()

"""
################## Different dataset size - RMNIST #####################
x = [100, 200, 300, 400, 500, 600]

dataset = "meta_dataset_complete"
partition = "None"
folders = [f"_saved_models/{dataset}/{partition}_data100",
           f"_saved_models/{dataset}/{partition}_data200",
           f"_saved_models/{dataset}/{partition}_data300",
           f"_saved_models/{dataset}/{partition}_data400",
           f"_saved_models/{dataset}/{partition}_data500",
           f"_saved_models/{dataset}/{partition}",
           ]
methods = ["fedavg", "fedavg_ft", "perfedavg", "ifca", "ifca_ft", "ditto", "fedrep", "proposed_c1"]

dict_methods_mean = {key: [] for key in methods}
dict_methods_std = {key: [] for key in methods}
for f in folders:
    file_dir = f + "/avg_comparison.csv"
    data = pd.read_csv(file_dir).set_index(['Unnamed: 0'])
    for i, name in enumerate(methods):
        dict_methods_mean[name].append(data.iloc[i][0])
        dict_methods_std[name].append(data.iloc[i][1])


fig = plt.figure(figsize=(10,7))
plt.rcParams.update({'font.size': 20})
plt.plot(x, dict_methods_mean["fedavg_ft"], label="FedAvg_FT", c='tab:pink', linewidth=2, marker='o')
#plt.fill_between(x, np.array(dict_methods_mean["fedavg_ft"])-np.array(dict_methods_std["fedavg_ft"]), np.array(dict_methods_mean["fedavg_ft"])+np.array(dict_methods_std["fedavg_ft"]), color='tab:pink', alpha=0.2)
plt.plot(x, dict_methods_mean["ifca_ft"], label="IFCA-FT", c="tab:purple", linewidth=2, marker='o')
#plt.fill_between(x, np.array(dict_methods_mean["ifca_ft"])-np.array(dict_methods_std["ifca_ft"]), np.array(dict_methods_mean["ifca_ft"])+np.array(dict_methods_std["ifca_ft"]), color='tab:purple', alpha=0.2)
plt.plot(x, dict_methods_mean["ditto"], label="Ditto", c="tab:orange", linewidth=2, marker='o')
#plt.fill_between(x, np.array(dict_methods_mean["ditto"])-np.array(dict_methods_std["ditto"]), np.array(dict_methods_mean["ditto"])+np.array(dict_methods_std["ditto"]), color='tab:orange', alpha=0.2)
plt.plot(x, dict_methods_mean["fedrep"], label="FedRep", c="tab:green", linewidth=2, marker='o')
#plt.fill_between(x, np.array(dict_methods_mean["fedrep"])-np.array(dict_methods_std["fedrep"]), np.array(dict_methods_mean["fedrep"])+np.array(dict_methods_std["fedrep"]), color='tab:green', alpha=0.2)
plt.plot(x, dict_methods_mean["perfedavg"], label="Per-FedAvg", c="tab:blue", linewidth=2, marker='o')
#plt.fill_between(x, np.array(dict_methods_mean["perfedavg"])-np.array(dict_methods_std["perfedavg"]), np.array(dict_methods_mean["perfedavg"])+np.array(dict_methods_std["perfedavg"]), color='tab:blue', alpha=0.2)
plt.plot(x, dict_methods_mean["proposed_c1"], label="CAFeMe", c="tab:red", linewidth=2, marker='o')
#plt.fill_between(x, np.array(dict_methods_mean["proposed_c1"])-np.array(dict_methods_std["proposed_c1"]), np.array(dict_methods_mean["proposed_c1"])+np.array(dict_methods_std["proposed_c1"]), color='tab:red', alpha=0.2)
plt.ylabel("Average accuracy")
plt.xlabel("Local data size")
plt.xticks(x)
plt.title("Meta-Dataset")
plt.legend(loc='upper left')
#plt.savefig('_saved_models/meta_dataset_complete/comparison_size.png')
plt.show()


