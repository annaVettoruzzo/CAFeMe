import torch
import copy
import numpy as np
from fedlab.utils.dataset.sampler import SubsetSampler

from datasets import get_dataloader
from utils import DEVICE, serialize_model_params, accuracy, evaluate_client


class FedAvgClient:
    def __init__(self, dataset, client_id, global_model, trainset, client_split, loss_fn, lr, batch_size=20, lr_ft=0.01):
        self.trainloader, self.valloader = get_dataloader(dataset, trainset, client_split, client_id, batch_size, val_ratio=0.2)
        self.iter_trainloader = iter(self.trainloader)

        self.loss_fn = loss_fn
        self.lr = lr
        self.lr_ft = lr_ft
        self.device = DEVICE

        self.local_model = copy.deepcopy(global_model)
        self.local_optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.lr)

    # -------------------------------------------------------------------
    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)

        return x.to(self.device), y.to(self.device)

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
        tot_acc = evaluate_client(global_model, self.valloader)
        return tot_acc

    # -------------------------------------------------------------------
    def perfl_eval(self, global_model, per_steps, only_fe=False):
        # Copy the model to avoid adapting the original one
        cmodel = copy.deepcopy(global_model)

        if only_fe: optimizer = torch.optim.SGD(cmodel.lin.parameters(), self.lr_ft)
        else: optimizer = torch.optim.SGD(cmodel.parameters(), self.lr_ft)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        test_acc = []
        for step in range(per_steps + 1):
            # Evaluate current model on the test data
            acc = evaluate_client(cmodel, self.valloader)
            test_acc.append(acc)

            # Adapt the model using training data
            x, y = self.get_data_batch()
            y_pred = cmodel(x)
            loss = self.loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 5 == 0:
                scheduler.step()

        return test_acc


class IFCAClient:
    def __init__(self, dataset, client_id, global_model, trainset, client_split, loss_fn, lr, batch_size=20, lr_ft=0.01):
        self.trainloader, self.valloader = get_dataloader(dataset, trainset, client_split, client_id, batch_size, val_ratio=0.2)
        self.iter_trainloader = iter(self.trainloader)

        self.loss_fn = loss_fn
        self.lr = lr
        self.lr_ft = lr_ft
        self.device = DEVICE

        self.local_model = [copy.deepcopy(model) for model in global_model] # this is a list of models
        self.local_optimizer = [torch.optim.SGD(model.parameters(), lr=self.lr) for model in self.local_model]

    # -------------------------------------------------------------------
    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)

        return x.to(self.device), y.to(self.device)

    # -------------------------------------------------------------------
    def compute_best_model(self, model=None):
        if model is None: model = self.local_model

        population_loss = [0]*len(model)
        for idx, m in enumerate(model):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                preds = m(images)
                loss = self.loss_fn(preds, labels)
                population_loss[idx] += loss.item()
        return np.argmin(population_loss)

    # -------------------------------------------------------------------
    def fit(self, global_model, adapt_steps):
        for idx, model in enumerate(global_model):
            self.local_model[idx].load_state_dict(model.state_dict())

        best_model_idx = self.compute_best_model()
        model, opt = self.local_model[best_model_idx], self.local_optimizer[best_model_idx]

        for i in range(adapt_steps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                preds = model(images)
                loss = self.loss_fn(preds, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                batch_loss.append(loss.item())

        self.local_model[best_model_idx].load_state_dict(dict(model.named_parameters()))
        serialized_list = [serialize_model_params(model) for model in self.local_model]

        return serialized_list, best_model_idx, np.mean(batch_loss)

    # -------------------------------------------------------------------
    def fl_eval(self, global_model):
        best_model_idx = self.compute_best_model(global_model)
        best_model = global_model[best_model_idx]

        tot_acc = evaluate_client(best_model, self.valloader)
        return tot_acc

    # -------------------------------------------------------------------
    def perfl_eval(self, global_model, per_steps):
        best_model_idx = self.compute_best_model(global_model)
        best_model = global_model[best_model_idx]

        # Copy the model to avoid adapting the original one
        cmodel = copy.deepcopy(best_model)

        optimizer = torch.optim.SGD(cmodel.parameters(), self.lr_ft)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        test_acc = []
        for step in range(per_steps + 1):
            # Evaluate current model on the test data
            acc = evaluate_client(cmodel, self.valloader)
            test_acc.append(acc)

            # Adapt the model using training data
            x, y = self.get_data_batch()
            y_pred = cmodel(x)
            loss = self.loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 5 == 0:
                scheduler.step()

        return test_acc