import torch
from torchvision import transforms

import numpy as np
import random

from utils import DEVICE, deserialize_model_params, aggregate_model_params
from modules import SimpleCNNModuleWithTE
from methods import MultimodalFL_Client
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
num_clients_per_round = 5

global_steps = 5000  # Num of communication rounds
adapt_steps = 5  # Num of local training rounds
per_steps = 50  # Num of personalization steps

########################### DATASET ###########################
dataset = "cifar10"
clients_training, clients_test = get_clients_id(num_clients, p_test=0.8)
trainset, shards_part = get_dataset(dataset, num_clients, transforms.ToTensor(), partition="shards", seed=seed)

# Models
global_model = SimpleCNNModuleWithTE(num_classes, "c1").to(DEVICE)

# init clients (every local model is initialized with the same global model)
clients = [MultimodalFL_Client(client_id=client_id,
                     trainset=trainset,
                     shards_part=shards_part,
                     global_model=global_model,
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     lr_in=lr_inner,
                     lr_out=lr_outer,
                     batch_size=batch_size)
    for client_id in range(num_clients)
]

# Train
for step in range(global_steps):
    selected_clients = random.sample(clients_training, num_clients_per_round)

    # client local training
    model_params_cache = []
    client_avg_loss = []
    for client_id in selected_clients:
        serialized_model_params, loss = clients[client_id].fit(global_model, adapt_steps)
        model_params_cache.append(serialized_model_params)
        client_avg_loss.append(loss)

    # aggregate
    aggregated_model_params = aggregate_model_params(model_params_cache)
    deserialize_model_params(global_model, aggregated_model_params)

    if (step + 1) % 50 == 0:
        print(f"Step: {step + 1}, loss: {np.mean(client_avg_loss):.5f}")

# Test
tot_acc = []
for client_id in clients_test:
    history = clients[client_id].perfl_eval(global_model, per_steps)
    tot_acc.append(history["eval"])
print(f"Accuracy: {np.mean(tot_acc, axis=0)}")








