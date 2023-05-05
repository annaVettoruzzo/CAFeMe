import random
import torch
import torchvision
import torchvision.transforms as tr
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset.sampler import SubsetSampler
from .femnist_dataset import FEMNIST, DatasetSplit


# -------------------------------------------------------------------
def get_dataset(dataset, num_clients=100, transforms=tr.ToTensor(), partition=None, seed=0):
    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=f"./data/{dataset}/", train=True, download=True, transform=transforms)
        if "unbalanced" in partition:
            partition_list = partition.split("_")
            client_split = CIFAR10Partitioner(trainset.targets, num_clients, balance=False, partition=partition_list[1], dir_alpha=0.3, unbalance_sgm=0.3, seed=seed)
        else:
            client_split = CIFAR10Partitioner(trainset.targets, num_clients, balance=None, partition=partition, num_shards=200, dir_alpha=0.3, seed=seed)
    elif dataset == "femnist":
        trainset = FEMNIST(train=True)
        all_client_split = trainset.get_client_dic()
        tmp_client_split = random.sample(list(all_client_split.items()), k=num_clients)
        client_split = dict((i, y) for i, (x, y) in enumerate(tmp_client_split))
    return trainset, client_split


# -------------------------------------------------------------------
def get_clients_id(num_clients, p_test=0.8):
    list_clients = list(range(num_clients))
    clients_training = random.sample(list_clients, int(num_clients * p_test))
    clients_test = list(set(list_clients) - set(clients_training))
    return clients_training, clients_test


# -------------------------------------------------------------------
def get_dataloader(dataset, trainset, client_split, client_id, batch_size, val_ratio=0.1):
    idxs = client_split[client_id]
    val_indexes = random.sample(list(idxs), int(val_ratio * len(idxs)))
    train_indexes = list(set(idxs) - set(val_indexes))
    trainloader = torch.utils.data.DataLoader(trainset, sampler=SubsetSampler(indices=train_indexes, shuffle=True), batch_size=batch_size, drop_last=True)
    valloader = torch.utils.data.DataLoader(trainset, sampler=SubsetSampler(indices=val_indexes, shuffle=True), batch_size=batch_size)
    return trainloader, valloader
