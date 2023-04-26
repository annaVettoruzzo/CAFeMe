import random
import torchvision
from fedlab.utils.dataset.partition import CIFAR10Partitioner


# -------------------------------------------------------------------
def get_dataset(dataset, num_clients, transforms, partition="shards", val_ratio=0.1, seed=0):
    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=f"./data/{dataset}/", train=True, download=True, transform=transforms)
        shards_part = CIFAR10Partitioner(trainset.targets, num_clients, balance=None, partition=partition, num_shards=200, seed=seed)
    return trainset, shards_part


# -------------------------------------------------------------------
def get_clients_id(num_clients, p_test=0.8):
    list_clients = list(range(num_clients))
    clients_training = random.sample(list_clients, int(num_clients * p_test))
    clients_test = list(set(list_clients) - set(clients_training))
    return clients_training, clients_test
