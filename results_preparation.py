import torch
import numpy as np
import random
import pickle
import pandas as pd
from pathlib import Path
from arguments import set_args

args = set_args()

# For reproducibility
seed = args["seed"]
torch.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

dataset = "rmnist"# args["dataset"] #rmnist, cifar10, femnist
partition = args["partition"]

if dataset == "cifar10":
    folders = [f"_saved_models/{dataset}/{partition}/seed0/",
               f"_saved_models/{dataset}/{partition}/seed1/",
               f"_saved_models/{dataset}/{partition}/seed2/",
               ]
else:
    folders = [f"_saved_models/{dataset}/seed0/",
               f"_saved_models/{dataset}/seed1/",
               f"_saved_models/{dataset}/seed2/",
               ]
PATH = Path(f"_saved_models/{dataset}/")
PATH.mkdir(parents=True, exist_ok=True)

methods = ["fedavg", "fedavg_ft", "perfedavg", "ifca", "ifca_ft", "ifca_sharing", "proposed_c1"]

df = pd.DataFrame(columns=["mean", "std"], index=methods)
for m in methods:
    avg_acc = []
    for f in folders:
        file_dir = f + f"acc_{m}.pkl"
        file = open(file_dir, 'rb')
        accuracy = pickle.load(file)
        if type(accuracy) == list: accuracy = np.mean(accuracy[-3:])
        file.close()
        avg_acc.append(np.mean(accuracy))

    df.loc[m] = [round(100*np.mean(avg_acc), 2), round(100*np.std(avg_acc), 2)]

print(df)
df.to_csv(PATH / f"avg_comparison.csv", encoding='utf-8')
