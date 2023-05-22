import torch
import random
import numpy as np
import os
from glob import glob
from collections import defaultdict
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as tr
from utils import load_tfrecord_images

default_transforms = tr.Compose([
    tr.ToPILImage(),
    tr.Resize((32, 32)),
    tr.ToTensor(),
])

# -------------------------------------------------------------------
special_transforms = tr.Compose([
    tr.Resize((32, 32)),
    tr.ToTensor(),
])

# -------------------------------------------------------------------
omniglot_transforms = tr.Compose([
    tr.ToPILImage(),
    tr.Grayscale(num_output_channels=3),
    tr.Resize((32, 32)),
    tr.ToTensor(),
    tr.Lambda(lambda v: 1. - v), # To be like the original omniglot dataset (black letter on white background)
])


# -------------------------------------------------------------------
class BaseDataset:
    def __init__(self, folder, transforms=tr.Compose([])):
        if 'imagenet' in folder or 'cifar' in folder:
            split = "train"
            dataset = torchvision.datasets.ImageFolder(f"{folder}/{split}", transform=transforms)

            # Group images by their class
            self.ds_dict = defaultdict(list)
            self.data, self.labels = [], []
            for img, c in dataset:
                self.ds_dict[c].append(img.numpy())
                self.data.append(img.numpy())
                self.labels.append(c)
        else:
            # List all the tfrecords filenames
            fnames = glob(os.path.join(folder, "*.tfrecords"))

            # Group images by their class
            self.ds_dict = defaultdict(list)
            self.data, self.labels = [], []
            for fname in fnames:
                images, c = load_tfrecord_images(fname)
                images = [transforms(img).numpy() for img in images]
                self.ds_dict[c] = images
                self.data += images
                self.labels += [c for _ in range(len(images))]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.data)


# -------------------------------------------------------------------
class AircraftDataset(BaseDataset):
    def __init__(self):
        folder = "./data/metadataset-records/aircraft/"
        super().__init__(folder, transforms=default_transforms)

# -------------------------------------------------------------------
class CUBirdsDataset(BaseDataset):
    def __init__(self):
        folder = "./data/metadataset-records/cu_birds/"
        super().__init__( folder, transforms=default_transforms)

# -------------------------------------------------------------------
class DtdDataset(BaseDataset):
    def __init__(self):
        folder = "./data/metadataset-records/dtd/"
        super().__init__(folder, transforms=default_transforms)

# -------------------------------------------------------------------
class TrafficSignDataset(BaseDataset):
    def __init__(self):
        folder = "./data/metadataset-records/traffic_sign/"
        super().__init__(folder, transforms=default_transforms)

# -------------------------------------------------------------------
class VggFlowerDataset(BaseDataset):
    def __init__(self):
        folder = "./data/metadataset-records/vgg_flower/"
        super().__init__(folder, transforms=default_transforms)

# -------------------------------------------------------------------
class OmniglotDataset(BaseDataset):
    def __init__(self):
        folder = "./data/metadataset-records/omniglot/"
        super().__init__(folder, transforms=omniglot_transforms)

# -------------------------------------------------------------------
class MiniImageNetDataset(BaseDataset):
    def __init__(self):
        folder = "./data/miniimagenet/"
        super().__init__(folder, transforms=special_transforms)

# -------------------------------------------------------------------
class CifarDataset(BaseDataset):
    def __init__(self):
        folder = "./data/cifarfs/"
        super().__init__(folder, transforms=special_transforms)

# -------------------------------------------------------------------
class MetaDataset(Dataset):
    def __init__(self, num_clients, num_data=600, num_classes=10, datasets=["aircraft", "cu_birds", "dtd", "traffic_sign", "vgg_flower", "omniglot", "miniimagenet", "cifar"]):
        dico = {
            "aircraft": AircraftDataset,
            "cu_birds": CUBirdsDataset,
            "dtd": DtdDataset,
            "traffic_sign": TrafficSignDataset,
            "vgg_flower": VggFlowerDataset,
            "omniglot": OmniglotDataset,
            "miniimagenet": MiniImageNetDataset,
            "cifar": CifarDataset,
        }


        self.num_clients = num_clients
        self.num_data = num_data
        self.num_classes = num_classes

        self.all_datasets = [DatasetGenerator() for name, DatasetGenerator in dico.items() if name in datasets]

        self.data, self.labels = [], []
        #self.dic_users = [[]] * self.num_clients
        self.dic_users = {}
        count = 0
        for i in range(self.num_clients):
            mode = random.choice(self.all_datasets)
            images, targets = self.sample_batch(mode)
            self.data += images
            self.labels += targets
            self.dic_users[i] = np.arange(len(images)) + count
            #user_info = {}
            #user_info["mode"] = mode
            #user_info["idxs"] = np.arange(len(images)) + count
            #user_info["targets"] = targets
            #self.dic_users[i] = user_info
            count += len(images)

    def sample_batch(self, mode):
        classes = list(mode.ds_dict.keys()) # All possible classes
        classes = random.sample(classes, self.num_classes)  # Randomly select n classes

        # Randomly map each selected class to a label in {0, ..., n-1}
        labels = random.sample(range(self.num_classes), self.num_classes)
        label_map = dict(zip(classes, labels))

        images, targets = [], []
        for c in classes:
            imgs = random.sample(mode.ds_dict[c], min(len(mode.ds_dict[c]), self.num_data//self.num_classes))  # FIXME: only ok for balanced setting
            images += imgs
            targets += [label_map[c] for _ in range(len(imgs))]
        return images, targets


    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        return self.dic_users


