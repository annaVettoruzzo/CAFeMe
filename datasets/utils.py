import random
import torch
import os
from glob import glob
from tfrecord.torch.dataset import TFRecordDataset
from cv2 import imdecode
import torchvision
import torchvision.transforms as tr
from torch.utils.data import Dataset
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset.sampler import SubsetSampler
from .femnist_dataset import FEMNIST
from .rmnist_dataset import RotatedMNIST, RotatedMNISTNewRotation


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
        trainset = FEMNIST()
        all_client_split = trainset.get_client_dic()
        tmp_client_split = random.sample(list(all_client_split.items()), k=num_clients)
        client_split = dict((i, y) for i, (x, y) in enumerate(tmp_client_split))
    elif dataset == "rmnist":
        trainset = RotatedMNIST(num_clients)
        client_split = trainset.get_client_dic()
    elif dataset == "rmnist_newrotation":
        trainset = RotatedMNISTNewRotation(num_clients)
        client_split = trainset.get_client_dic()
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


# -------------------------------------------------------------------
def load_tfrecord_images(fpath):
    dataset = TFRecordDataset(fpath, None, {"image": "byte", "label": "int"})
    dataset = list(dataset)
    label = dataset[0]["label"][0]
    images = [imdecode(dico["image"], -1) for dico in dataset]
    return images, label


# -------------------------------------------------------------------
class BaseDataset:
    def __init__(self, folder, transforms=tr.Compose([])):
        # List all the tfrecords filenames
        fnames = glob(os.path.join(folder, "*.tfrecords"))

        # Group images by their class
        self.data, self.labels = [], []
        for fname in fnames:
            images, c = load_tfrecord_images(fname)
            images = [transforms(img).numpy() for img in images]
            self.data += images
            self.labels += [c for _ in range(len(images))]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.data)
