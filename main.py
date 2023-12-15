import copy

import torch
from torchvision import transforms
import numpy as np
import random
from pathlib import Path
from utils import DEVICE, train, train_ifca, evaluate_fl
from methods import MultimodalFL_Client, PerFedAvg_Client, Ditto_Client, FedRep_Client, FedAvgClient, IFCAClient
from modules import BaseHeadSplit
from datasets import get_dataset, get_clients_id
from arguments import set_args

args = set_args()

# For reproducibility
seed = args["seed"]
torch.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

dataset = args["dataset"]  # Possible choices: 'rmnist', 'cifar10', 'femnist'
partition = args["partition"]

# For saving models
PATH = Path(f"_saved_models/{dataset}_complete/{partition}/seed{seed}")
PATH.mkdir(parents=True, exist_ok=True)
print(PATH)

########################### DATASET ###########################
trainset, client_split = get_dataset(args, transforms=transforms.ToTensor())
clients_training, clients_test = get_clients_id(args["num_clients"], args["p_val"])

########################### TRAINING PROPOSED ###########################
global_model = args["model_proposed"].to(DEVICE)
clients = [MultimodalFL_Client(args["dataset"], client_id, global_model, trainset, client_split, args["loss_fn"], args["lr_inner"], args["lr_outer"], args["batch_size"], args["lr_ft_pfl"])
           for client_id in range(args["num_clients"])]

# train
global_model = train(global_model, clients, clients_training, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"])
#torch.save(global_model.state_dict(), PATH / "proposed_c1")
#global_model.load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/proposed_c1"))
# test
accuracy_proposed = evaluate_fl(global_model, clients, clients_test, args["per_steps"], save=PATH / "acc_proposed_c1")
print(f"Accuracy proposed: {accuracy_proposed}")

########################### TRAINING PER-FEDAVG ###########################
perfedavg_model = args["model"].to(DEVICE)
clients = [PerFedAvg_Client(args["dataset"], client_id, perfedavg_model, trainset, client_split, args["loss_fn"], args["lr_inner"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]
# train
perfedavg_model = train(perfedavg_model, clients, clients_training, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"])
torch.save(perfedavg_model.state_dict(), PATH / "per_fedavg")
#perfedavg_model.load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/per_fedavg"))
# test
accuracy_perfedavg = evaluate_fl(perfedavg_model, clients, clients_test, args["per_steps"], save=PATH / "acc_perfedavg")
print(f"Accuracy PerFedAvg: {accuracy_perfedavg}")

########################### TRAINING FEDAVG ###########################
fedavg_model = args["model"].to(DEVICE)
clients = [FedAvgClient(args["dataset"], client_id, fedavg_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]

# train
fedavg_model = train(fedavg_model, clients, clients_training, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"])
torch.save(fedavg_model.state_dict(), PATH / "fedavg")
#fedavg_model.load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/fedavg"))

# test
accuracy_fedavg = evaluate_fl(fedavg_model, clients, clients_test, fine_tuning=False, save=PATH / "acc_fedavg")
print(f"Accuracy FedAvg: {accuracy_fedavg}")

# (set only_fe=True if aggregate only feature extractor)
#fedavg_model = train(fedavg_model, clients, clients_training, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"], only_fe=True)
#torch.save(fedavg_model.state_dict(), PATH / "fedavg_only_fe")
#accuracy_fedavg = evaluate_fl(fedavg_model, clients, clients_test, fine_tuning=True, only_fe=True, save=PATH / "acc_fedavg_only_fe")

########################### TRAINING FEDAVG-FT ###########################
accuracy_fedavg_ft = evaluate_fl(fedavg_model, clients, clients_test, args["per_steps"], save=PATH / "acc_fedavg_ft")
print(f"Accuracy FedAvg-FT: {accuracy_fedavg_ft}")

###################### TRAINING IFCA - w/o weights sharing ###########################
n_models = args["n_models_ifca"]
ifca_model = [args["model"].to(DEVICE) for _ in range(n_models)]
clients = [IFCAClient(args["dataset"], client_id, ifca_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]

# train
ifca_model = train_ifca(ifca_model, clients, clients_training, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"])
for i in range(len(ifca_model)): torch.save(ifca_model[i].state_dict(), PATH / f"ifca{i}")
#for i in range(len(ifca_model)) : ifca_model[i].load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/ifca{i}"))

# test
accuracy_ifca = evaluate_fl(ifca_model, clients, clients_test, fine_tuning=False, save=PATH / "acc_ifca")
print(f"Accuracy IFCA: {accuracy_ifca}")

########################### TRAINING IFCA-FT ###########################
accuracy_ifca_ft = evaluate_fl(ifca_model, clients, clients_test, args["per_steps"], save=PATH / "acc_ifca_ft")
print(f"Accuracy IFCA-FT: {accuracy_ifca_ft}")


###################### TRAINING IFCA - with weights sharing ###########################
n_models = args["n_models_ifca"]
ifca_sharing_model = [args["model"].to(DEVICE) for _ in range(n_models)]
clients = [IFCAClient(args["dataset"], client_id, ifca_sharing_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args["batch_size"])
           for client_id in range(args["num_clients"])]
# train
ifca_sharing_model = train_ifca(ifca_sharing_model, clients, clients_training, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"], weight_sharing=True)
for i in range(len(ifca_sharing_model)): torch.save(ifca_sharing_model[i].state_dict(), PATH / f"ifca_sharing{i}")
#for i in range(len(ifca_sharing_model)) : ifca_sharing_model[i].load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/ifca_sharing{i}"))

# test
accuracy_ifca_sharing = evaluate_fl(ifca_sharing_model, clients, clients_test, fine_tuning=False, save=PATH / "acc_ifca_sharing")
print(f"Accuracy IFCA sharing weights: {accuracy_ifca_sharing}")

###################### TRAINING DITTO ###########################
ditto_model = args["model"].to(DEVICE)
clients = [Ditto_Client(args["dataset"], client_id, ditto_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args['mu'], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]

# train
ditto_model = train(ditto_model, clients, clients_training, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"])
torch.save(ditto_model.state_dict(), PATH / "ditto")
#ditto_model.load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/ditto"))

# test
accuracy_ditto = evaluate_fl(ditto_model, clients, clients_test, args["per_steps"], save=PATH / "acc_ditto")
print(f"Accuracy Ditto: {accuracy_ditto}")

###################### TRAINING FedRep ###########################
fedrep_model = args["model"].to(DEVICE)
head = copy.deepcopy(fedrep_model.fc)
fedrep_model.fc = torch.nn.Identity()
fedrep_model = BaseHeadSplit(fedrep_model, head)
clients = [FedRep_Client(args["dataset"], client_id, fedrep_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]

# train
fedrep_model = train(fedrep_model, clients, clients_training, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"])
torch.save(fedrep_model.state_dict(), PATH / "fedrep")
#fedrep_model.load_state_dict(torch.load(f"_saved_models/{dataset}/seed{seed}/fedrep"))

# test
accuracy_fedrep = evaluate_fl(fedrep_model, clients, clients_test, args["per_steps"], save=PATH / "acc_fedrep")
print(f"Accuracy FedRep: {accuracy_fedrep}")

