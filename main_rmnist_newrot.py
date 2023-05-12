import torch
from torchvision import transforms
import numpy as np
import random
from pathlib import Path
from utils import DEVICE, train, train_ifca, evaluate_fl
from methods import MultimodalFL_Client, PerFedAvg_Client, FedAvgClient, IFCAClient
from datasets import get_dataset, get_clients_id
from arguments import set_args

args = set_args()

# For reproducibility
seed = args["seed"]
torch.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

dataset = args["dataset"] #rmnist, cifar10, femnist
partition = args["partition"]

# For saving models
if dataset == "cifar10": PATH = Path(f"_saved_models/{dataset}/{partition}/seed{seed}")
elif dataset == "rmnist" and args["newrotation"]: PATH = Path(f"_saved_models/{dataset}_newrotation/seed{seed}")
else: PATH = Path(f"_saved_models/{dataset}/seed{seed}")
PATH.mkdir(parents=True, exist_ok=True)
print(PATH)

########################### DATASET ###########################
# train
trainset, client_split = get_dataset(dataset, num_clients=args["num_clients"], transforms=transforms.ToTensor(), partition=partition, seed=seed)
clients_training, clients_test = get_clients_id(args["num_clients"], args["p_val"])

# test
name = dataset + "_newrotation"
num_clients_test = int(args["num_clients"]*round(1-args["p_val"], 2))
testset, client_split_test = get_dataset(name, num_clients=num_clients_test, transforms=transforms.ToTensor())

########################### TRAINING PROPOSED ###########################
global_model = args["model_proposed"].to(DEVICE)
global_model.load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/proposed_c1"))

clients = [MultimodalFL_Client(args["dataset"], client_id, global_model, trainset, client_split, args["loss_fn"], args["lr_inner"], args["lr_outer"], args["batch_size"], args["lr_ft_pfl"])
           for client_id in range(args["num_clients"])]

clients_test = [MultimodalFL_Client(args["dataset"], client_id, global_model, testset, client_split_test, args["loss_fn"], args["lr_inner"], args["lr_outer"], args["batch_size"], args["lr_ft_pfl"])
           for client_id in range(num_clients_test)]

# test
accuracy_proposed = evaluate_fl(global_model, clients_test, np.arange(len(clients_test)).tolist(), args["per_steps"], save=PATH / "acc_proposed_c1")
print(f"Accuracy proposed: {accuracy_proposed[-1]}")


########################### TRAINING PER-FEDAVG ###########################
perfedavg_model = args["model"].to(DEVICE)
perfedavg_model.load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/per_fedavg"))

clients = [PerFedAvg_Client(args["dataset"], client_id, perfedavg_model, trainset, client_split, args["loss_fn"], args["lr_inner"], args["lr_outer"], args["batch_size"], args["lr_ft_pfl"])
           for client_id in range(args["num_clients"])]

clients_test = [PerFedAvg_Client(args["dataset"], client_id, perfedavg_model, testset, client_split_test, args["loss_fn"], args["lr_inner"], args["lr_outer"], args["batch_size"], args["lr_ft_pfl"])
           for client_id in range(num_clients_test)]
# test
accuracy_perfedavg = evaluate_fl(perfedavg_model, clients_test, np.arange(len(clients_test)).tolist(), args["per_steps"], save=PATH / "acc_perfedavg")
print(f"Accuracy PerFedAvg: {accuracy_perfedavg[-1]}")


########################### TRAINING FEDAVG ###########################
fedavg_model = args["model"].to(DEVICE)
fedavg_model.load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/fedavg"))

clients = [FedAvgClient(args["dataset"], client_id, fedavg_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]

clients_test = [FedAvgClient(args["dataset"], client_id, fedavg_model, testset, client_split_test, args["loss_fn"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(num_clients_test)]

# test
accuracy_fedavg = evaluate_fl(fedavg_model, clients_test, np.arange(len(clients_test)).tolist(), fine_tuning=False, save=PATH / "acc_fedavg")
print(f"Accuracy FedAvg: {accuracy_fedavg}")

########################### TRAINING FEDAVG-FT ###########################
accuracy_fedavg_ft = evaluate_fl(fedavg_model, clients_test, np.arange(len(clients_test)).tolist(), args["per_steps"], save=PATH / "acc_fedavg_ft")
print(f"Accuracy FedAvg-FT: {accuracy_fedavg_ft[-1]}")

###################### TRAINING IFCA - w/o weights sharing ###########################
n_models = 3
ifca_model = [args["model"].to(DEVICE) for _ in range(n_models)]
for i in range(len(ifca_model)) : ifca_model[i].load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/ifca{i}"))

clients = [IFCAClient(args["dataset"], client_id, ifca_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]

clients_test = [IFCAClient(args["dataset"], client_id, ifca_model, testset, client_split_test, args["loss_fn"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(num_clients_test)]

# test
accuracy_ifca = evaluate_fl(ifca_model, clients_test, np.arange(len(clients_test)).tolist(), fine_tuning=False, save=PATH / "acc_ifca")
print(f"Accuracy IFCA: {accuracy_ifca}")


########################### TRAINING IFCA-FT ###########################
accuracy_ifca_ft = evaluate_fl(ifca_model, clients_test, np.arange(len(clients_test)).tolist(), args["per_steps"], save=PATH / "acc_ifca_ft")
print(f"Accuracy IFCA-FT: {accuracy_ifca_ft[-1]}")


###################### TRAINING IFCA - with weights sharing ###########################
n_models = 3
ifca_sharing_model = [args["model"].to(DEVICE) for _ in range(n_models)]
for i in range(len(ifca_sharing_model)) : ifca_sharing_model[i].load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/ifca_sharing{i}"))

clients = [IFCAClient(args["dataset"], client_id, ifca_sharing_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args["batch_size"])
           for client_id in range(args["num_clients"])]

clients_test = [IFCAClient(args["dataset"], client_id, ifca_sharing_model, testset, client_split_test, args["loss_fn"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(num_clients_test)]
# test
accuracy_ifca_sharing = evaluate_fl(ifca_sharing_model, clients_test, np.arange(len(clients_test)).tolist(), fine_tuning=False, save=PATH / "acc_ifca_sharing")
print(f"Accuracy IFCA sharing weights: {accuracy_ifca_sharing}")