import random
import torchvision
import torchvision.transforms as tr
from torch.utils.data import Dataset, Subset, BatchSampler, DataLoader
from sklearn.model_selection import train_test_split
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset.sampler import SubsetSampler
from .femnist_dataset import FEMNIST
from .rmnist_dataset import RotatedMNIST, RotatedMNISTNewRotation
from .meta_dataset import MetaDataset


# -------------------------------------------------------------------
def get_dataset(args, transforms=tr.ToTensor()):
    dataset = args["dataset"]
    partition = args["partition"]
    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=f"./data/{dataset}/", train=True, download=True, transform=transforms)
        if "unbalanced" in partition:
            partition_list = partition.split("_")
            client_split = CIFAR10Partitioner(trainset.targets, args["num_clients"], balance=False, partition=partition_list[1], dir_alpha=0.3, unbalance_sgm=0.3, seed=args["seed"])
        else:
            client_split = CIFAR10Partitioner(trainset.targets, args["num_clients"], balance=None, partition=partition, num_shards=200, dir_alpha=0.3, seed=args["seed"])
    elif dataset == "femnist":
        trainset = FEMNIST(args["num_clients"])
        client_split = trainset.get_client_dic()
    elif dataset == "femnist" and partition == "shards":
        trainset = FEMNIST(args["num_clients"])
        client_split = CIFAR10Partitioner(trainset.targets, args["num_clients"], balance=None, partition=partition, num_shards=200, dir_alpha=0.3, seed=args["seed"])
    elif dataset == "rmnist":
        trainset = RotatedMNIST(args["num_clients"], args["k_clusters"])
        if partition == "shards" or partition == "dirichlet":
            client_split = CIFAR10Partitioner(trainset.targets, args["num_clients"], balance=None, partition=partition, num_shards=200, dir_alpha=0.3, seed=args["seed"])
        else:
            client_split = trainset.get_client_dic()
    elif dataset == "rmnist_newrotation":
        trainset = RotatedMNISTNewRotation(args["num_clients"])
        client_split = trainset.get_client_dic()
    elif dataset == "meta_dataset":
        trainset = MetaDataset(args["num_clients"], args["num_data"], args["num_classes"], args["datasets"])
        client_split = trainset.get_client_dic()
    return trainset, client_split


# -------------------------------------------------------------------
def get_clients_id(num_clients, p_test=0.8):
    list_clients = list(range(num_clients))
    clients_training = random.sample(list_clients, int(num_clients * p_test))
    clients_test = list(set(list_clients) - set(clients_training))
    return clients_training, clients_test


# -------------------------------------------------------------------
def get_dataloader(dataset_name, trainset, client_split, client_id, batch_size, val_ratio=0.1):
    """
    if dataset_name == "meta_dataset":
        user = trainset.dic_users[client_id]
        # Split the dataset into train and validation indices while ensuring at least one sample per class in both sets
        #train_indices, val_indices = train_test_split(user["idxs"], test_size=val_ratio, stratify=user["targets"], random_state=0)
        train_indices, val_indices = [], []
        for class_label in np.unique(user["targets"]):
            class_indices = [idx for idx, target in zip(user["idxs"], user["targets"]) if target == class_label]
            class_train_indices, class_val_indices = train_test_split(class_indices, test_size=int(val_ratio * len(class_indices)), random_state=0)
            train_indices.extend(class_train_indices)
            val_indices.extend(class_val_indices)
        # Create the train and validation subsets using the train_indices and val_indices
        train_dataset = Subset(trainset, train_indices)
        val_dataset = Subset(trainset, val_indices)
        # Create sampler to make sure each s´batch has at least one sample per class
        batch_sampler_train = OneSamplePerClassBatchSampler(train_dataset, batch_size)
        batch_sampler_val = OneSamplePerClassBatchSampler(val_dataset, batch_size)
        # Create dataloader
        trainloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train)
        valloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val)
    else:
    """
    idxs = client_split[client_id]
    val_indexes = random.sample(list(idxs), int(val_ratio * len(idxs)))
    train_indexes = list(set(idxs) - set(val_indexes))
    trainloader = DataLoader(trainset, sampler=SubsetSampler(indices=train_indexes, shuffle=True), batch_size=batch_size, drop_last=True)
    valloader = DataLoader(trainset, sampler=SubsetSampler(indices=val_indexes, shuffle=True), batch_size=batch_size)
    return trainloader, valloader


# -------------------------------------------------------------------
class OneSamplePerClassBatchSampler(BatchSampler):
    """ It ensures each batch contains at least one sample per class """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_indices = self._get_class_indices()

    def _get_class_indices(self):
        class_indices = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def __iter__(self):
        batch = []
        remaining_indices = list(range(len(self.dataset)))

        for indices in self.class_indices.values():
            if not indices:
                self.class_indices = self._get_class_indices()

        for indices in self.class_indices.values():
            selected_index = indices.pop(0)
            remaining_indices.remove(selected_index)
            batch.append(selected_index)
        while remaining_indices:
            remaining_indices_len = len(remaining_indices)
            if remaining_indices_len >= self.batch_size-len(self.class_indices):
                batch.extend(remaining_indices[:self.batch_size-len(self.class_indices)])
                remaining_indices = remaining_indices[self.batch_size:]
            else:
                batch.extend(remaining_indices)
                remaining_indices = []
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size

