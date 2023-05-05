import json
import torch
from torch.utils.data import Dataset
import logging
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List
from torch.utils.data.dataset import ConcatDataset


class FemnistDataset(Dataset):
    def __init__(self, client_id: int, client_str: str, data: list,
                 targets: list):
        """get `Dataset` for femnist dataset
         Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): image data list
            targets (list): image class target list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.data = data
        self.targets = targets
        self._process_data_target()

    def _process_data_target(self):
        """process client's data and target
        """
        self.data = torch.tensor(self.data,
                                 dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

logging.getLogger().setLevel(logging.INFO)


class PickleDataset:
    """Splits LEAF generated datasets and creates individual client partitions."""

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name for dataset of PickleDataset Object
            data_root (str): path for data saving root.
                             Default to None and will be modified to the datasets folder in FedLab: "fedlab-benchmarks/datasets"
            pickle_root (str): path for pickle dataset file saving root.
                             Default to None and will be modified to Path(__file__).parent / "pickle_datasets"
        """
        self.dataset_name = dataset_name
        self.data_root = Path("../data")
        self.pickle_root = Path("../data")

    def create_pickle_dataset(self):
        # for train file data
        train_path = self.data_root / self.dataset_name / "data/train"
        original_train_datasets = sorted(list(train_path.glob("**/*.json")))
        self._read_process_json_data(dataset_type="train", paths_to_json=original_train_datasets)

        # for test file data
        test_path = self.data_root / self.dataset_name / "data/test"
        original_test_datasets = sorted(list(test_path.glob("**/*.json")))
        self._read_process_json_data(dataset_type="test", paths_to_json=original_test_datasets)

    def get_dataset_pickle(self, dataset_type: str, client_id: int = None):
        """load pickle dataset file for `dataset_name` `dataset_type` data based on client with client_id
        Args:
            dataset_type (str): Dataset type {train, test}
            client_id (int): client id. Defaults to None, which means get all_dataset pickle
        Raises:
            FileNotFoundError: No such file or directory {pickle_root}/{dataset_name}/{dataset_type}/{dataset_type}_{client_id}.pickle
        Returns:
            if there is no pickle file for `dataset`, throw FileNotFoundError, else return responding dataset
        """
        # check whether to get all datasets
        if client_id is None:
            pickle_files_path = self.pickle_root / self.dataset_name / dataset_type
            dataset_list = []
            for file in list(pickle_files_path.glob("**/*.pkl")):
                dataset_list.append(pickle.load(open(file, 'rb')))
            dataset = ConcatDataset(dataset_list)
        else:
            pickle_file = self.pickle_root / self.dataset_name / dataset_type / f"{dataset_type}_{client_id}.pkl"
            dataset = pickle.load(open(pickle_file, 'rb'))
        return dataset

    def _read_process_json_data(self, dataset_type: str, paths_to_json: List[Path]):
        """read and process LEAF generated datasets to responding Dataset
        Args:
            dataset_type (str): Dataset type {train, test}
            paths_to_json (PathLike): Path to LEAF JSON files containing dataset.
        """
        user_count = 0

        logging.info(f"processing {self.dataset_name} {dataset_type} data to dataset in pickle file")

        for path_to_json in paths_to_json:
            with open(path_to_json, "r") as json_file:
                json_file = json.load(json_file)
                users_list = sorted(json_file["users"])
                num_users = len(users_list)
                for user_idx, user_str in enumerate(tqdm(users_list)):
                    self._process_user(json_file, user_count + user_idx, user_str, dataset_type)
            user_count += num_users
        logging.info(f"""
                    Complete processing {self.dataset_name} {dataset_type} data to dataset in pickle file! 
                    Located in {(self.pickle_root / self.dataset_name / dataset_type).resolve()}. 
                    All users number is {user_count}.
                    """)

    def _process_user(self, json_file: Dict[str, Any], user_idx: str, user_str: str, dataset_type: str):
        """Creates and saves partition for user
        Args:
            json_file (Dict[str, Any]): JSON file containing user data
            user_idx (str): User ID (counter) in string format
            user_str (str): Original User ID
            dataset_type (str): Dataset type {train, test}
        """
        data = json_file["user_data"][user_str]["x"]
        label = json_file["user_data"][user_str]["y"]
        if self.dataset_name == "femnist":
            dataset = FemnistDataset(client_id=user_idx,
                                     client_str=user_str,
                                     data=data,
                                     targets=label)
        else:
            raise ValueError("Invalid dataset:", self.dataset_name)

        # save_dataset_pickle
        save_dir = self.pickle_root / self.dataset_name / dataset_type
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f"{dataset_type}_{str(user_idx)}.pkl", "wb") as save_file:
            pickle.dump(dataset, save_file)

    def get_data_json(self, dataset_type: str):
        """ Read .json file from ``data_dir``
        This is modified by [LEAF/models/utils/model_utils.py]
        https://github.com/TalwalkarLab/leaf/blob/master/models/utils/model_utils.py
        Args:
            dataset_type (str): Dataset type {train, test}
        Returns:
            clients name dict mapping keys to id, groups list for each clients, a dict data mapping keys to client
        """
        groups = []
        all_data = []
        client_name2data = dict()

        data_dir = self.data_root / self.dataset_name / "data" / dataset_type
        files = list(data_dir.glob("**/*.json"))
        for f in files:
            with open(f, 'r') as inf:
                cdata = json.load(inf)
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            client_name2data.update(cdata['user_data'])
            # get all data
            for key, item in cdata["user_data"].items():
                for text in item['x']:
                    if self.dataset_name == 'sent140':
                        all_data.append(text[4])

        # generate clients_id_str - client_id_index map
        clients_name = list(sorted(client_name2data.keys()))
        clients_id = list(range(len(clients_name)))
        client_id2name = dict(zip(clients_id, clients_name))

        return client_id2name, groups, client_name2data, all_data


    def get_built_vocab(self, vocab_save_root: str = None):
        """load vocab file for `dataset` to get Vocab based on selected client and data in current directory
        Args:
            vocab_save_root (str): string of vocab saving root path, which corresponds to the save_root param in `build_vocab.py()`
                            Default to None, which will be modified to Path(__file__).parent / "dataset_vocab"
        Returns:
            if there is no built vocab file for `dataset`, return None, else return Vocab
        """
        save_root = Path(__file__).parent / "nlp_utils/dataset_vocab" if vocab_save_root is None else vocab_save_root
        vocab_file_path = save_root / f'{self.dataset_name}_vocab.pkl'

        if not vocab_file_path.exists():
            logging.warning(f"""
                            No built vocab file {self.dataset_name}_vocab.pkl for {self.dataset_name} dataset in {save_root.resolve()}!
                            We will build it with default vocab_limit_size 50000 to generate it firstly!
                            You can also build it by running {BASE_DIR}/leaf/nlp_utils/build_vocab.sh
                            """)
            self.build_vocab()  # use default vocab saving path and vocab_limit_size 50000
        vocab_file = open(vocab_file_path, 'rb')
        vocab = pickle.load(vocab_file)
        return vocab


if __name__ == '__main__':

    pdataset = PickleDataset(dataset_name="femnist")

    pdataset.create_pickle_dataset()