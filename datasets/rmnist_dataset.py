import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class RotatedMNIST(Dataset):
    def __init__(self, num_clients=100, data_per_client=2400, k_clusters=4):
        super(RotatedMNIST, self).__init__()
        self.num_clients = num_clients
        self.data_per_client = data_per_client
        self.k_clusters = k_clusters

        data = datasets.MNIST(root='RMNIST', train=True, download=True, transform=transforms.ToTensor())
        self.data = data.data
        self.labels = data.targets

        assert (num_clients // k_clusters) * data_per_client == len(self.data)
        self.data_indices, self.group_assign = self.assign_groups(num_clients, k_clusters)

        self.dic_users = {}
        for i in range(self.num_clients):
            self.dic_users[i] = self.data_indices[i]

    def __getitem__(self, index):
        img, target, group_id = self.data[index], self.labels[index], self.group_assign[index]
        img = img / 255.0
        img = img.unsqueeze(0)
        img = torch.rot90(img, k=int(group_id + 1), dims=(1, 2))
        img = img.reshape(-1, 28, 28)
        return img, target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        return self.dic_users

    def assign_groups(self, num_clients, k_clusters):
        data_indices = []
        group_assign = []
        clients_per_cluster = num_clients//k_clusters
        data_idxs = np.arange(len(self.data))
        np.random.shuffle(data_idxs) #shuffle to randomly sampling
        chunks = np.split(data_idxs, k_clusters)
        for k_i in range(k_clusters):
            indices_per_client = np.split(chunks[k_i], clients_per_cluster)
            data_indices += indices_per_client
            group_assign += [k_i for _ in range(len(chunks[k_i]))]
        data_indices = np.array(data_indices)
        group_assign = np.array(group_assign)
        return data_indices, group_assign
