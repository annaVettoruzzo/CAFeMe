import torch
from utils import DEVICE
import argparse
from modules import SimpleCNNModuleWithTE, SimpleCNNModule, SimpleFNNModuleWithTE, SimpleFNNModule


def set_args():
    args_dict = {}
    """
    # CMD line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int, help='Seed')
    parser.add_argument('dataset', type=str, choices=["cifar10", "femnist", "rmnist"], help='Dataset')
    parser.add_argument('--partition', type=str, choices=["shards", "dirichlet", "unbalanced_iid"], default=None, help='Partition for cifar10')
    args = parser.parse_args()

    args_dict["dataset"] = args.dataset
    args_dict["seed"] = args.seed
    args_dict["partition"] = args.partition
    """
    args_dict["dataset"] = "femnist"
    args_dict["seed"] = 0
    args_dict["partition"] = None

    if args_dict["dataset"] == "cifar10":
        args_dict["p_val"] = 0.8
        args_dict["num_clients"] = 100
        args_dict["num_classes"] = 10
        args_dict["batch_size"] = 50
        args_dict["num_clients_per_round"] = 5

        args_dict["lr_inner"] = 0.05
        args_dict["lr_outer"] = 0.001
        args_dict["loss_fn"] = torch.nn.CrossEntropyLoss().to(DEVICE)

        args_dict["global_steps"] = 1000  # Num of communication rounds
        args_dict["adapt_steps"] = 5  # Num of local training rounds
        args_dict["per_steps"] = 50  # Num of personalization steps

        args_dict["model_proposed"] = SimpleCNNModuleWithTE(conv_dim=[3, 64, 64, 64], dense_dim=[1024, 576, 576], n_classes=10, modulation="c1")
        args_dict["model"] = SimpleCNNModule(conv_dim=[3, 64, 64, 64], dense_dim=[1024, 576, 576], n_classes=10)

    if args_dict["dataset"] == "femnist":
        args_dict["p_val"] = 0.8
        args_dict["num_clients"] = 200
        args_dict["num_classes"] = 62
        args_dict["batch_size"] = 50
        args_dict["num_clients_per_round"] = 5

        args_dict["lr_inner"] = 0.05
        args_dict["lr_outer"] = 0.001
        args_dict["loss_fn"] = torch.nn.CrossEntropyLoss().to(DEVICE)

        args_dict["global_steps"] = 2000  # Num of communication rounds
        args_dict["adapt_steps"] = 5  # Num of local training rounds
        args_dict["per_steps"] = 50  # Num of personalization steps

        args_dict["model_proposed"] = SimpleCNNModuleWithTE(conv_dim=[1, 64, 64, 64], dense_dim=[576, 576, 576], n_classes=62, modulation="c1")
        args_dict["model"] = SimpleCNNModule(conv_dim=[1, 64, 64, 64], dense_dim=[576, 576, 576], n_classes=62)

    if args_dict["dataset"] == "rmnist":
        args_dict["p_val"] = 0.8
        args_dict["num_clients"] = 100
        args_dict["num_classes"] = 10
        args_dict["batch_size"] = 30
        args_dict["num_clients_per_round"] = 5

        args_dict["lr_inner"] = 0.05
        args_dict["lr_outer"] = 0.001
        args_dict["loss_fn"] = torch.nn.CrossEntropyLoss().to(DEVICE)

        args_dict["global_steps"] = 1000  # Num of communication rounds
        args_dict["adapt_steps"] = 5  # Num of local training rounds
        args_dict["per_steps"] = 50  # Num of personalization steps

        args_dict["model_proposed"] = SimpleFNNModuleWithTE(conv_dim=[1, 32, 64], dense_dim=[1024, 512], n_classes=10, modulation="c1")
        args_dict["model"] = SimpleFNNModule(conv_dim=[1, 32, 64], dense_dim=[1024, 512], n_classes=10)

    return args_dict

