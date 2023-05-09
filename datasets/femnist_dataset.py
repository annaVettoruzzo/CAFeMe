import json
import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import torch


class FEMNIST(Dataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    def __init__(self):
        super(FEMNIST, self).__init__()

        train_clients, train_groups, train_data_temp = read_data("./data/femnist/data/train")

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
        self.label = train_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = np.array([img])
        return torch.from_numpy((0.5-img)/0.5).float(), target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        return self.dic_users


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


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