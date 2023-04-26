import torch
from torchvision import transforms
import numpy as np
import random
from pathlib import Path
from utils import DEVICE, train, evaluate_perfl, evaluate_fl
from modules import SimpleCNNModuleWithTE, SimpleCNNModule
from methods import MultimodalFL_Client, PerFedAvg_Client, FedAvgClient
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
accuracy_proposed = evaluate_perfl(global_model, clients, clients_test, per_steps, save=PATH / "acc_proposed_c1")
print(f"Accuracy proposed: {accuracy_proposed[-1]}")

########################### TRAINING PER-FEDAVG ###########################
perfedavg_model = SimpleCNNModule(num_classes).to(DEVICE)
clients = [PerFedAvg_Client(client_id, trainset, shards_part, perfedavg_model, loss_fn, lr_inner, lr_outer, batch_size)
           for client_id in range(num_clients)]
# train
perfedavg_model = train(perfedavg_model, clients, clients_training, num_clients_per_round, adapt_steps, global_steps)
torch.save(perfedavg_model.state_dict(), PATH / "per_fedavg")
# test
accuracy_perfedavg = evaluate_perfl(perfedavg_model, clients, clients_test, per_steps, save=PATH / "acc_perfedavg")
print(f"Accuracy PerFedAvg: {accuracy_perfedavg[-1]}")

########################### TRAINING FEDAVG ###########################
fedavg_model = SimpleCNNModule(num_classes).to(DEVICE)
clients = [FedAvgClient(client_id, trainset, shards_part, fedavg_model, loss_fn, lr_outer, batch_size)
           for client_id in range(num_clients)]
# train
fedavg_model = train(fedavg_model, clients, clients_training, num_clients_per_round, adapt_steps, global_steps)
torch.save(fedavg_model.state_dict(), PATH / "fedavg")
# test
testset = get_dataset(dataset, test=True)
accuracy_fedavg = evaluate_fl(fedavg_model, testset, batch_size, save=PATH / "acc_fedavg")
print(f"Accuracy FedAvg: {accuracy_fedavg}")
