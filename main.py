import torch
from torchvision import transforms
import numpy as np
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
from utils import DEVICE, train, train_ifca, evaluate_fl
from modules import SimpleCNNModuleWithTE, SimpleCNNModule
from methods import MultimodalFL_Client, PerFedAvg_Client, FedAvgClient, IFCAClient
from datasets import get_dataset, get_clients_id

# For reproducibility
seed = 0
torch.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Parameters
num_clients = 100
num_classes = 10
batch_size = 50

lr_inner = 0.05
lr_outer = 0.001
loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
num_clients_per_round = 5

global_steps = 1000  # Num of communication rounds
adapt_steps = 5  # Num of local training rounds
per_steps = 50  # Num of personalization steps

dataset = "cifar10"
clients_training, clients_test = get_clients_id(num_clients, p_test=0.8)
trainset, shards_part = get_dataset(dataset, num_clients, transforms.ToTensor(), partition="shards", seed=seed)

# For saving models
PATH = Path(f"_saved_models/{dataset}/seed{seed}")
PATH.mkdir(parents=True, exist_ok=True)
print(PATH)

########################### TRAINING PROPOSED ###########################
global_model = SimpleCNNModuleWithTE(num_classes, "c1").to(DEVICE)
clients = [MultimodalFL_Client(client_id, trainset, shards_part, global_model, loss_fn, lr_inner, lr_outer, batch_size)
           for client_id in range(num_clients)]
# train
global_model = train(global_model, clients, clients_training, num_clients_per_round, adapt_steps, global_steps)
torch.save(global_model.state_dict(), PATH / "proposed_c1")
# test
accuracy_proposed = evaluate_fl(global_model, clients, clients_test, per_steps, save=PATH / "acc_proposed_c1")
print(f"Accuracy proposed: {accuracy_proposed[-1]}")

########################### TRAINING PER-FEDAVG ###########################
perfedavg_model = SimpleCNNModule(num_classes).to(DEVICE)
clients = [PerFedAvg_Client(client_id, trainset, shards_part, perfedavg_model, loss_fn, lr_inner, lr_outer, batch_size)
           for client_id in range(num_clients)]
# train
perfedavg_model = train(perfedavg_model, clients, clients_training, num_clients_per_round, adapt_steps, global_steps)
torch.save(perfedavg_model.state_dict(), PATH / "per_fedavg")
# test
accuracy_perfedavg = evaluate_fl(perfedavg_model, clients, clients_test, per_steps, save=PATH / "acc_perfedavg")
print(f"Accuracy PerFedAvg: {accuracy_perfedavg[-1]}")

########################### TRAINING FEDAVG ###########################
fedavg_model = SimpleCNNModule(num_classes).to(DEVICE)
clients = [FedAvgClient(client_id, trainset, shards_part, fedavg_model, loss_fn, lr_outer, batch_size)
           for client_id in range(num_clients)]
# train
fedavg_model = train(fedavg_model, clients, clients_training, num_clients_per_round, adapt_steps, global_steps)
torch.save(fedavg_model.state_dict(), PATH / "fedavg")
# test
accuracy_fedavg = evaluate_fl(fedavg_model, clients, clients_test, fine_tuning=False, save=PATH / "acc_fedavg")
print(f"Accuracy FedAvg: {accuracy_fedavg}")

########################### TRAINING FEDAVG-FT ###########################
accuracy_fedavg_ft = evaluate_fl(fedavg_model, clients, clients_test, per_steps, save=PATH / "acc_fedavg_ft")
print(f"Accuracy FedAvg-FT: {accuracy_fedavg_ft}")


###################### TRAINING IFCA - w/o weights sharing ###########################
n_models = 3
ifca_model = [SimpleCNNModule(num_classes).to(DEVICE) for _ in range(n_models)]
clients = [IFCAClient(client_id, trainset, shards_part, ifca_model, loss_fn, lr_outer, batch_size)
           for client_id in range(num_clients)]
# train
ifca_model = train_ifca(ifca_model, clients, clients_training, num_clients_per_round, adapt_steps, global_steps)
for i in range(len(ifca_model)): torch.save(ifca_model[i].state_dict(), PATH / f"ifca{i}")
# test
accuracy_ifca = evaluate_fl(ifca_model, clients, clients_test, fine_tuning=False, save=PATH / "acc_ifca")
print(f"Accuracy IFCA: {accuracy_ifca}")


###################### TRAINING IFCA - with weights sharing ###########################
n_models = 3
ifca_sharing_model = [SimpleCNNModule(num_classes).to(DEVICE) for _ in range(n_models)]
clients = [IFCAClient(client_id, trainset, shards_part, ifca_sharing_model, loss_fn, lr_outer, batch_size)
           for client_id in range(num_clients)]
# train
ifca_sharing_model = train_ifca(ifca_sharing_model, clients, clients_training, num_clients_per_round, adapt_steps, global_steps, weight_sharing=True)
for i in range(len(ifca_sharing_model)): torch.save(ifca_sharing_model[i].state_dict(), PATH / f"ifca_sharing{i}")
# test
accuracy_ifca_sharing = evaluate_fl(ifca_sharing_model, clients, clients_test, fine_tuning=False, save=PATH / "acc_ifca_sharinf")
print(f"Accuracy IFCA sharing weights: {accuracy_ifca_sharing}")


###################### TRAINING G-FML ###########################

###################### EVALUATE ###########################
accuracy_fedavg = pd.read_pickle(PATH/"acc_fedavg.pkl")
accuracy_fedavg_ft = pd.read_pickle(PATH/"acc_fedavg_ft.pkl")
accuracy_perfedavg = pd.read_pickle(PATH/"acc_perfedavg.pkl")
accuracy_ifca = pd.read_pickle(PATH/"acc_ifca.pkl")
accuracy_ifca_sharing = pd.read_pickle(PATH/"acc_ifca_sharing.pkl")
accuracy_proposed = pd.read_pickle(PATH/"acc_proposed_c1.pkl")

plt.plot(accuracy_fedavg, label="fedavg")
plt.plot(accuracy_fedavg_ft, label="fedavg-ft")
plt.plot(accuracy_perfedavg, label="per-fedavg")
plt.plot(accuracy_ifca, label="ifca")
plt.plot(accuracy_ifca_sharing, label="ifca-sharing")
plt.plot(accuracy_proposed, label="proposed")
plt.legend()
plt.show()
