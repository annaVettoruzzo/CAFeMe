import torch
import json
import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import random


# -------------------------------------------------------------------
class FEMNIST(Dataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    def __init__(self, num_clients=200):
        train_clients, _, train_data_temp = read_data("./data/femnist/data/train")
        train_clients = random.sample(train_clients, k=num_clients)

        self.dic_users = {}
        train_data_x = []
        train_data_y = []
        for i in range(len(train_clients)):
            self.dic_users[i] = set()
            l = len(train_data_x)
            cur_x = train_data_temp[train_clients[i]]['x']
            cur_y = train_data_temp[train_clients[i]]['y']
            for j in range(len(cur_x)):
                self.dic_users[i].add(j + l)
                train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                train_data_y.append(cur_y[j])

        self.data = train_data_x
        self.targets = train_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = np.array([img])
        return torch.from_numpy((0.5-img)/0.5).float(), target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        return self.dic_users


# -------------------------------------------------------------------
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# -------------------------------------------------------------------
def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir):
    '''
    parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)


    return train_clients, train_groups, train_data


def shards_partition(targets, num_clients, num_shards):
    """Non-iid partition used in FedAvg `paper <https://arxiv.org/abs/1602.05629>`_ Now implemented for FEMNIST.
    Code from https://github.com/SMILELab-FL/FedLab.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        num_shards (int): Number of shards in partition.

    Returns:
        dict: ``{ client_id: indices}``.
    """

    num_samples = len(targets)

    size_shard = int(num_samples / num_shards)

    shards_per_client = int(num_shards / num_clients)

    indices = np.arange(num_samples)
    # sort sample indices according to labels
    indices_targets = np.vstack((indices, targets))
    indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
    # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
    sorted_indices = indices_targets[0, :]

    # permute shards idx, and slice shards_per_client shards for each client
    rand_perm = np.random.permutation(num_shards)
    num_client_shards = np.ones(num_clients) * shards_per_client
    # sample index must be int
    num_cumsum = np.cumsum(num_client_shards).astype(int)
    # shard indices for each client
    client_shards_dict = split_indices(num_cumsum, rand_perm)

    # map shard idx to sample idx for each client
    client_dict = dict()
    for cid in range(num_clients):
        shards_set = client_shards_dict[cid]
        current_indices = [
            sorted_indices[shard_id * size_shard: (shard_id + 1) * size_shard]
            for shard_id in shards_set]
        client_dict[cid] = np.concatenate(current_indices, axis=0)

    return client_dict


def split_indices(num_cumsum, rand_perm):
    """Splice the sample index list given number of each client.
    Code from https://github.com/SMILELab-FL/FedLab.

    Args:
        num_cumsum (np.ndarray): Cumulative sum of sample number for each client.
        rand_perm (list): List of random sample index.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    client_indices_pairs = [(cid, idxs) for cid, idxs in
                            enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    client_dict = dict(client_indices_pairs)
    return client_dict