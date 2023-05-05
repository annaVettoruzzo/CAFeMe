import torch
from utils import DEVICE


def set_args(dataset):
    args_dict = {}
    if dataset == "cifar10":
        args_dict["dataset"] = dataset
        args_dict["p_val"] = 0.8
        args_dict["num_clients"] = 100
        args_dict["num_classes"] = 10
        args_dict["batch_size"] = 50
        args_dict["partition"] = "unbalanced_iid" #shards, dirichlet, #unbalanced_iid
        args_dict["num_clients_per_round"] = 5

        args_dict["lr_inner"] = 0.05
        args_dict["lr_outer"] = 0.001
        args_dict["loss_fn"] = torch.nn.CrossEntropyLoss().to(DEVICE)

        args_dict["global_steps"] = 1000  # Num of communication rounds
        args_dict["adapt_steps"] = 5  # Num of local training rounds
        args_dict["per_steps"] = 50  # Num of personalization steps

        args_dict["conv_dim"] = [3, 64, 64, 64]
        args_dict["dense_dim"] = [1024, 576, 576]

    if dataset == "femnist":
        args_dict["dataset"] = dataset
        args_dict["p_val"] = 0.8
        args_dict["num_clients"] = 200
        args_dict["num_classes"] = 62
        args_dict["batch_size"] = 50
        args_dict["partition"] = None
        args_dict["num_clients_per_round"] = 5

        args_dict["lr_inner"] = 0.05
        args_dict["lr_outer"] = 0.001
        args_dict["loss_fn"] = torch.nn.CrossEntropyLoss().to(DEVICE)

        args_dict["global_steps"] = 1000  # Num of communication rounds
        args_dict["adapt_steps"] = 5  # Num of local training rounds
        args_dict["per_steps"] = 50  # Num of personalization steps

        args_dict["conv_dim"] = [1, 16, 16, 16]
        args_dict["dense_dim"] = [144, 128, 128]



    return args_dict

