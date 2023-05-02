import random
import torch
import torchvision
import torchvision.transforms as tr
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset.sampler import SubsetSampler

# -------------------------------------------------------------------
def get_dataset(dataset, num_clients=100, transforms=tr.ToTensor(), partition="shards", seed=0):
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


# -------------------------------------------------------------------
def get_dataloader(trainset, shards_part, client_id, batch_size, val_ratio=0.1):
    # for testing
    val_indexes = random.sample(list(shards_part[client_id]), int(val_ratio * len(shards_part[client_id])))
    train_indexes = list(set(shards_part[client_id]) - set(val_indexes))

    trainloader = torch.utils.data.DataLoader(trainset, sampler=SubsetSampler(indices=train_indexes, shuffle=True), batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(trainset, sampler=SubsetSampler(indices=val_indexes, shuffle=True), batch_size=batch_size)

    return trainloader, testloader
