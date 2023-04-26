import torch
import copy
import numpy as np
from fedlab.utils.dataset.sampler import SubsetSampler

from utils import DEVICE, serialize_model_params, accuracy


class FedAvgClient:
    def __init__(self, client_id, trainset, shards_part, global_model, loss_fn, lr, batch_size=20):
        self.trainloader = torch.utils.data.DataLoader(trainset, sampler=SubsetSampler(indices=shards_part[client_id], shuffle=True), batch_size=batch_size)

        self.loss_fn = loss_fn
        self.lr = lr
        self.device = DEVICE

        self.local_model = copy.deepcopy(global_model)
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr)

    # -------------------------------------------------------------------
    def fit(self, global_model, adapt_steps):
        self.local_model.load_state_dict(global_model.state_dict())

        for i in range(adapt_steps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                preds = self.local_model(images)
                loss = self.loss_fn(preds, labels)
                self.local_optimizer.zero_grad()
                loss.backward()
                self.local_optimizer.step()

                batch_loss.append(loss.item())

        return serialize_model_params(self.local_model), np.mean(batch_loss)

    # -------------------------------------------------------------------
    def fl_eval(self, global_model):
        tot_acc = []
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Inference
            outputs = global_model(images)
            eval_acc = accuracy(outputs, labels)
            tot_acc.append(eval_acc)

        return np.mean(tot_acc)
