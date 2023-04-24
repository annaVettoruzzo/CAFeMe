import torch, random
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from rich.console import Console

from data import preprocess, get_client_id_indices
from modules import SimpleCNNModuleWithTE
from methods import FL_Client
from utils import DEVICE

# For reproducibility
seed = 0
torch.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

########################### PARAMETERS ###########################
lr_inner = 0.01
lr_outer = 0.001
batch_size = 40  # Batch size of client local dataset

global_steps = 1000  # Num of communication rounds
local_steps = 4  # Num of local training rounds
pers_steps = 50  # Num of personalization rounds (while in evaluation phase)

dataset = "cifar"
classes = 10
client_num = 50
client_num_per_round = 10  # Num of clients that participating training at each communication round
p = 0.9  # Percentage of training clients

########################### DATASET ###########################
print("Dataset preprocessing")
preprocess(dataset, client_num, p, classes)
clients_training, clients_eval, client_num_in_total = get_client_id_indices(dataset)

########################### MODELS ###########################
global_model = SimpleCNNModuleWithTE(classes, modulation='c1').to(DEVICE)

# init clients (every local model is initialized with the same global model)
clients = [
    FL_Client(
        client_id=client_id,
        dataset=dataset,
        global_model=global_model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        lr_in=lr_inner,
        lr_out=lr_outer,
        batch_size=batch_size
    )
    for client_id in range(client_num_in_total)
]

# training
for step in range(global_steps):
    # select clients
    selected_clients = random.sample(clients_training, client_num_per_round)

    # client local training
    model_params_cache = []
    for client_id in selected_clients:
        serialized_model_params = clients[client_id].fit(global_model, local_steps)
        model_params_cache.append(serialized_model_params)

    # aggregate model parameters
    aggregated_model_params = Aggregators.fedavg_aggregate(model_params_cache)
    SerializationTool.deserialize_model(global_model, aggregated_model_params)

    if (step + 1) % 50 == 0:
        print(f"Step: {step + 1}")

# eval
loss_before, loss_after, acc_before, acc_after = [], [], [], []
for client_id in clients_eval:
    stats = clients[client_id].pfl_eval(global_model, pers_steps)

    loss_before.append(stats["loss_before"])
    loss_after.append(stats["loss_after"])
    acc_before.append(stats["acc_before"])
    acc_after.append(stats["acc_after"])

print("=" * 20, "RESULTS", "=" * 20)
print(f"loss_before_pers: {(sum(loss_before) / len(loss_before)):.4f}")
print(f"acc_before_pers: {(sum(acc_before) * 100.0 / len(acc_before)):.2f}%")
print(f"loss_after_pers: {(sum(loss_after) / len(loss_after)):.4f}")
print(f"acc_after_pers: {(sum(acc_after) * 100.0 / len(acc_after)):.2f}%")