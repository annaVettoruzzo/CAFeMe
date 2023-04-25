import warnings
warnings.filterwarnings("ignore")
import torch
import random
import numpy as np
import os
import pickle
import json
import shutil
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple
from collections import Counter
from fedlab.utils.dataset.slicing import noniid_slicing
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from .datasets import MNISTDataset, CIFARDataset
from torch.utils.data import Dataset
from utils import IMAGE_SIZE

DATASET = {
    "mnist": (MNIST, MNISTDataset),
    "cifar": (CIFAR10, CIFARDataset),
}


MEAN = {
    "mnist": (0.1307,),
    "cifar": (0.4914, 0.4822, 0.4465),
}

STD = {
    "mnist": (0.3015,),
    "cifar": (0.2023, 0.1994, 0.2010),
}


def preprocess(dataset, client_num_in_total, fraction, classes):
    dataset_dir = Path(f"data/{dataset}")
    pickles_dir = Path(f"data/{dataset}/pickles")

    num_train_clients = int(client_num_in_total * fraction)
    num_test_clients = client_num_in_total - num_train_clients

    transform = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
         transforms.ToTensor()]
    )

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    if os.path.isdir(pickles_dir):
        shutil.rmtree(pickles_dir, ignore_errors=True)
    os.mkdir(pickles_dir)

    ori_dataset, target_dataset = DATASET[dataset]
    trainset = ori_dataset(dataset_dir, train=True, download=True, transform=transform)
    testset = ori_dataset(dataset_dir, train=False, transform=transform)

    num_classes = 10 if classes <= 0 else classes
    all_trainsets, trainset_stats = randomly_alloc_classes(
        ori_dataset=trainset,
        target_dataset=target_dataset,
        num_clients=num_train_clients,
        num_classes=num_classes
    )
    all_testsets, testset_stats = randomly_alloc_classes(
        ori_dataset=testset,
        target_dataset=target_dataset,
        num_clients=num_test_clients,
        num_classes=num_classes
    )

    all_datasets = all_trainsets + all_testsets

    for client_id, d in enumerate(all_datasets):
        with open(pickles_dir / (str(client_id) + ".pkl"), "wb") as f:
            pickle.dump(d, f)
    with open(pickles_dir / "seperation.pkl", "wb") as f:
        pickle.dump(
            {
                "train": [i for i in range(num_train_clients)],
                "test": [i for i in range(num_train_clients, client_num_in_total)],
                "total": client_num_in_total,
            },
            f,
        )
    with open(dataset_dir / "all_stats.json", "w") as f:
        json.dump({"train": trainset_stats, "test": testset_stats}, f)


def randomly_alloc_classes(
    ori_dataset: Dataset,
    target_dataset: Dataset,
    num_clients: int,
    num_classes: int,
    transform=None,
    target_transform=None,
) -> Tuple[List[Dataset], Dict[str, Dict[str, int]]]:
    dict_users = noniid_slicing(ori_dataset, num_clients, num_clients * num_classes)
    stats = {}
    for i, indices in dict_users.items():
        targets_numpy = np.array(ori_dataset.targets)
        stats[f"client {i}"] = {"x": 0, "y": {}}
        stats[f"client {i}"]["x"] = len(indices)
        stats[f"client {i}"]["y"] = Counter(targets_numpy[indices].tolist())
    datasets = []
    for indices in dict_users.values():
        datasets.append(
            target_dataset(
                [ori_dataset[i] for i in indices],
                transform=transform,
                target_transform=target_transform,
            )
        )
    return datasets, stats

