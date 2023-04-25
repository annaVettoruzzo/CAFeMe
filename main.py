import torch
from torchvision import transforms
import numpy as np
import random
from pathlib import Path
from utils import DEVICE, train
from modules import SimpleCNNModuleWithTE, SimpleCNNModule
from methods import MultimodalFL_Client, PerFedAvg_Client
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
tot_acc = []
for client_id in clients_test:
    history = clients[client_id].perfl_eval(global_model, per_steps)
    tot_acc.append(history["eval"])

print(f"Accuracy: {np.mean(tot_acc, axis=0)}")

########################### TRAINING PER-FEDAVG ###########################
perfedavg_model = SimpleCNNModule(num_classes).to(DEVICE)
clients = [PerFedAvg_Client(client_id, trainset, shards_part, perfedavg_model, loss_fn, lr_inner, lr_outer, batch_size)
           for client_id in range(num_clients)]
# train
perfedavg_model = train(perfedavg_model, clients, clients_training, num_clients_per_round, adapt_steps, global_steps)
torch.save(perfedavg_model.state_dict(), PATH / "per_fedavg")

# test
tot_acc = []
for client_id in clients_test:
    history = clients[client_id].perfl_eval(perfedavg_model, per_steps)
    tot_acc.append(history["eval"])

print(f"Accuracy: {np.mean(tot_acc, axis=0)}")

