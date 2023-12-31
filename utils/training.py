import torch
import random
import numpy as np
from pathlib import Path
from utils import find_indices, deserialize_model_params, deserialize_specific_model_params, aggregate_model_params
from .common import reset_weights, evaluate_fl

# -------------------------------------------------------------------
def train(global_model, clients, clients_training, num_clients_per_round, adapt_steps, global_steps, only_fe=False):
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

        if only_fe:
            reset_weights(global_model, name_layer="lin")

        if (step + 1) % 50 == 0:
            print(f"Step: {step + 1}, loss: {np.mean(client_avg_loss):.5f}", end="\t\r")

    return global_model


# -------------------------------------------------------------------
def train_ifca(global_model, clients, clients_training, num_clients_per_round, adapt_steps, global_steps, weight_sharing=False):
    for step in range(global_steps):
        selected_clients = random.sample(clients_training, num_clients_per_round)

        # client local training
        model_params_cache = []
        model_idxs = []
        client_avg_loss = []
        for client_id in selected_clients:
            serialized_model_params, updated_model_idxs, loss = clients[client_id].fit(global_model, adapt_steps)
            model_params_cache.append(serialized_model_params)
            model_idxs.append(updated_model_idxs)
            client_avg_loss.append(loss)

        # aggregate at cluster level
        for i, model in enumerate(global_model):
            indexes = find_indices(model_idxs, i)
            if not indexes: continue
            model_params_indexes = [model_params_cache[p][i] for p in indexes]
            aggregated_model_params = aggregate_model_params(model_params_indexes)
            deserialize_model_params(global_model[i], aggregated_model_params)

        if weight_sharing:
            all_model_params_cache = [model_params_cache[i][idx] for i, idx in enumerate(model_idxs)]
            aggregated_model_params = aggregate_model_params(all_model_params_cache)
            for model in global_model:
                deserialize_specific_model_params(model, aggregated_model_params, "cnn")

        if (step + 1) % 50 == 0:
            print(f"Step: {step + 1}, loss: {np.mean(client_avg_loss):.5f}", end="\t\r")

    return global_model

# -------------------------------------------------------------------
def train_and_eval(global_model, clients, clients_training, clients_test, num_clients_per_round, adapt_steps, global_steps, test_steps, fine_tuning=True, only_fe=False, save_dir=None):
    path_dir = Path(save_dir)
    path_dir.mkdir(parents=True, exist_ok=True)

    test_accuracy = []

    for step in range(global_steps):
        selected_clients = random.sample(clients_training, num_clients_per_round)

        # Evaluation
        if step == 0 or (step+1) % 10 == 0:
            test_acc = evaluate_fl(global_model, clients, clients_test, test_steps, fine_tuning)
            try:
                if len(test_acc) > 3:
                    test_acc = np.mean(test_acc[-3:])
                    test_accuracy.append(test_acc)
            except:
                test_accuracy.append(test_acc)

            dir_name = path_dir / f"step{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': global_model.state_dict(),
                'test_acc': test_acc,
            }, dir_name)

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

        if only_fe:
            reset_weights(global_model, name_layer="lin")

        if (step+1) % 50 == 0:
            print(f"Step: {step + 1}, loss: {np.mean(client_avg_loss):.5f}", end="\t\r")

    return global_model, test_accuracy


# -------------------------------------------------------------------
def train_and_eval_ifca(global_model, clients, clients_training, clients_test, num_clients_per_round, adapt_steps, global_steps, test_steps, fine_tuning=True, weight_sharing=False, save_dir=None):
    path_dir = Path(save_dir)
    path_dir.mkdir(parents=True, exist_ok=True)

    test_accuracy = []

    for step in range(global_steps):
        selected_clients = random.sample(clients_training, num_clients_per_round)

        # client local training
        model_params_cache = []
        model_idxs = []
        client_avg_loss = []
        for client_id in selected_clients:
            serialized_model_params, updated_model_idxs, loss = clients[client_id].fit(global_model, adapt_steps)
            model_params_cache.append(serialized_model_params)
            model_idxs.append(updated_model_idxs)
            client_avg_loss.append(loss)

        # aggregate at cluster level
        for i, model in enumerate(global_model):
            indexes = find_indices(model_idxs, i)
            if not indexes: continue
            model_params_indexes = [model_params_cache[p][i] for p in indexes]
            aggregated_model_params = aggregate_model_params(model_params_indexes)
            deserialize_model_params(global_model[i], aggregated_model_params)

        if weight_sharing:
            all_model_params_cache = [model_params_cache[i][idx] for i, idx in enumerate(model_idxs)]
            aggregated_model_params = aggregate_model_params(all_model_params_cache)
            for model in global_model:
                deserialize_specific_model_params(model, aggregated_model_params, "cnn")

        if step == 0 or (step+1) % 10 == 0:
            test_acc = evaluate_fl(global_model, clients, clients_test, test_steps, fine_tuning)
            try:
                if len(test_acc) > 3:
                    test_acc = np.mean(test_acc[-3:])
                    test_accuracy.append(test_acc)
            except:
                test_accuracy.append(test_acc)

            dir_name = path_dir / f"step{step + 1}.pt"
            checkpoint = {'step': step+1, 'model_state_dict': [], 'test_acc': test_acc}
            for i, model in enumerate(global_model):
                checkpoint['model_state_dict'].append(model.state_dict())
            torch.save(checkpoint, dir_name)

        if (step + 1) % 50 == 0:
            print(f"Step: {step + 1}, loss: {np.mean(client_avg_loss):.5f}", end="\t\r")

    return global_model, test_accuracy