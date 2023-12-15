import torch
from torchvision import transforms
import numpy as np
import random
import copy
from pathlib import Path
from utils import DEVICE, train, train_ifca, evaluate_fl, train_and_eval, train_and_eval_ifca
from methods import MultimodalFL_Client, PerFedAvg_Client, Ditto_Client, FedRep_Client
from modules import BaseHeadSplit
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
PATH = Path(f"_saved_models/{dataset}/{partition}_steps_new/seed{seed}")
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
global_model, proposed_acc = train_and_eval(global_model, clients, clients_training, clients_test, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"], args["per_steps"],
                                            save_dir=PATH/"proposed_c1")


########################### TRAINING PER-FEDAVG ###########################
perfedavg_model = args["model"].to(DEVICE)
clients = [PerFedAvg_Client(args["dataset"], client_id, perfedavg_model, trainset, client_split, args["loss_fn"], args["lr_inner"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]

# train
perfedavg_model, perfedavg_acc = train_and_eval(perfedavg_model, clients, clients_training, clients_test, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"], args["per_steps"],
                                                save_dir=PATH/"perfedavg")

###################### TRAINING DITTO ###########################
ditto_model = args["model"].to(DEVICE)
clients = [Ditto_Client(args["dataset"], client_id, ditto_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args['mu'], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]

# train
ditto_model, ditto_acc = train_and_eval(ditto_model, clients, clients_training, clients_test, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"], args["per_steps"],
                                        save_dir=PATH/"ditto")

###################### TRAINING FedRep ###########################
fedrep_model = args["model"].to(DEVICE)
head = copy.deepcopy(fedrep_model.fc)
fedrep_model.fc = torch.nn.Identity()
fedrep_model = BaseHeadSplit(fedrep_model, head)
clients = [FedRep_Client(args["dataset"], client_id, fedrep_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]

fedrep_model, fedrep_acc = train_and_eval(fedrep_model, clients, clients_training, clients_test, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"], args["per_steps"],
                              save_dir=PATH/"fedrep")


"""

########################### TRAINING FEDAVG ###########################
fedavg_model = args["model"].to(DEVICE)
clients = [FedAvgClient(args["dataset"], client_id, fedavg_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]

# train
fedavg_model, fedavg_acc = train_and_eval(fedavg_model, clients, clients_training, clients_test, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"], args["per_steps"],
                                          fine_tuning=False, save_dir=PATH/"fedavg")

########################### TRAINING FEDAVG-FT ###########################
path_dir = Path(PATH/"fedavg_ft")
path_dir.mkdir(parents=True, exist_ok=True)
fedavf_ft_accuracy = []
for step in range(args["global_steps"]):
    if (step + 1) % 100 == 0:
        checkpoint = torch.load(PATH/"fedavg"/ f"step{step + 1}.pt")
        fedavg_model.load_state_dict(checkpoint['model_state_dict'])

        test_acc = evaluate_fl(fedavg_model, clients, clients_test, args["per_steps"], fine_tuning=True)
        test_acc = np.mean(test_acc[-1:])
        fedavf_ft_accuracy.append(test_acc)

        dir_name = PATH/"fedavg_ft" / f"step{step + 1}.pt"
        torch.save({
            'step': step + 1,
            'model_state_dict': fedavg_model.state_dict(),
            'test_acc': test_acc,
        }, dir_name)

###################### TRAINING IFCA - w/o weights sharing ###########################
n_models = args["n_models_ifca"]
ifca_model = [args["model"].to(DEVICE) for _ in range(n_models)]
clients = [IFCAClient(args["dataset"], client_id, ifca_model, trainset, client_split, args["loss_fn"], args["lr_outer"], args["batch_size"], args["lr_ft"])
           for client_id in range(args["num_clients"])]

# train
ifca_model, ifca_acc = train_and_eval_ifca(ifca_model, clients, clients_training, clients_test, args["num_clients_per_round"], args["adapt_steps"], args["global_steps"], args["per_steps"],
                                           fine_tuning=False, save_dir=PATH/"ifca")

########################### TRAINING IFCA-FT ###########################
path_dir = Path(PATH/"ifca_ft")
path_dir.mkdir(parents=True, exist_ok=True)
ifca_ft_acc = []
for step in range(args["global_steps"]):
    if (step + 1) % 100 == 0:
        checkpoint = torch.load(PATH/"ifca"/ f"step{step + 1}.pt")
        for i in range(len(ifca_model)):
            ifca_model[i].load_state_dict(checkpoint['model_state_dict'][i])

        test_acc = evaluate_fl(ifca_model, clients, clients_test, args["per_steps"], fine_tuning=True)
        ifca_ft_acc.append(test_acc[-1])

        dir_name = PATH/"ifca_ft" / f"step{step + 1}.pt"
        checkpoint = {'step': step + 1, 'model_state_dict': [], 'test_acc': np.mean(test_acc[-3:])}
        for i, model in enumerate(ifca_model):
            checkpoint['model_state_dict'].append(model.state_dict())
        torch.save(checkpoint, dir_name)
        
"""