import torch
import random
import numpy as np
from utils import deserialize_model_params, aggregate_model_params


def train(global_model, clients, clients_training, num_clients_per_round, adapt_steps, global_steps):
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
            print(f"Step: {step + 1}, loss: {np.mean(client_avg_loss):.5f}", end="\t\r")

    return global_model
