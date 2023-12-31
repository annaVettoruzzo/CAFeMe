import torch
import copy
from collections import OrderedDict

from datasets import get_dataloader
from utils import DEVICE, func_call, serialize_model_params, evaluate_client, PerturbedGradientDescent


class MultimodalFL_Client:
    """ This is the proposed CAFeMe approach. """
    def __init__(self, dataset, client_id, global_model, trainset, client_split, loss_fn, lr_in, lr_out=0.001, batch_size=20, lr_ft=0.05):
        self.trainloader, self.valloader = get_dataloader(dataset, trainset, client_split, client_id, batch_size, val_ratio=0.2)
        self.iter_trainloader = iter(self.trainloader)

        self.loss_fn = loss_fn
        self.lr_in = lr_in
        self.lr_out = lr_out
        self.lr_ft = lr_ft
        self.device = DEVICE

        self.local_model = copy.deepcopy(global_model)
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr_out)

    # -------------------------------------------------------------------
    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)

        return x.to(self.device), y.to(self.device)

    # -------------------------------------------------------------------
    def get_eval_data_batch(self, size_ratio=6):
        x_test, y_test = self.get_data_batch()
        for i in range(size_ratio - 1):
            x, y = self.get_data_batch()
            x_test, y_test = torch.cat((x_test, x), dim=0), torch.cat((y_test, y), dim=0)

        return x_test.to(self.device), y_test.to(self.device)

    # -------------------------------------------------------------------
    def adapt(self, params_dict, data_batch):
        x, y = data_batch
        y_pred = func_call(self.local_model, params_dict, x)
        inner_loss = self.loss_fn(y_pred, y)

        grads = torch.autograd.grad(inner_loss, params_dict.values())
        adapted_params_dict = {name: w - self.lr_in * w_grad for (name, w), w_grad in zip(params_dict.items(), grads)}

        return adapted_params_dict

    # -------------------------------------------------------------------
    def get_adapted_parameters(self, adapt_steps):
        data_batch = self.get_data_batch()
        phi = self.adapt(dict(self.local_model.named_parameters()), data_batch)
        for _ in range(adapt_steps - 1):
            data_batch = self.get_data_batch()
            phi = self.adapt(phi, data_batch)
        return phi

    # -------------------------------------------------------------------
    def fit(self, global_model, adapt_steps):
        self.local_model.load_state_dict(global_model.state_dict())

        phi = self.get_adapted_parameters(adapt_steps)

        x_2, y_2 = self.get_eval_data_batch()
        y_pred = func_call(self.local_model, phi, x_2)
        loss = self.loss_fn(y_pred, y_2)

        self.local_optimizer.zero_grad()
        loss.backward()
        self.local_optimizer.step()

        return serialize_model_params(self.local_model), loss.item()

    # -------------------------------------------------------------------
    def perfl_eval(self, global_model, per_steps):
        # Copy the model to avoid adapting the original one
        cmodel = copy.deepcopy(global_model)

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



class PerFedAvg_Client:
    def __init__(self, dataset, client_id, global_model, trainset, client_split, loss_fn, lr_in, lr_out=0.001, batch_size=20, lr_ft=0.01):
        self.trainloader, self.valloader = get_dataloader(dataset, trainset, client_split, client_id, batch_size, val_ratio=0.2)
        self.iter_trainloader = iter(self.trainloader)

        self.loss_fn = loss_fn
        self.lr_in = lr_in
        self.lr_out = lr_out
        self.lr_ft = lr_ft
        self.device = DEVICE

        self.local_model = copy.deepcopy(global_model)
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr_out)

    # -------------------------------------------------------------------
    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)

        return x.to(self.device), y.to(self.device)

    # -------------------------------------------------------------------
    def get_eval_data_batch(self, size_ratio=6):
        x_test, y_test = self.get_data_batch()
        for i in range(size_ratio - 1):
            x, y = self.get_data_batch()
            x_test, y_test = torch.cat((x_test, x), dim=0), torch.cat((y_test, y), dim=0)

        return x_test.to(self.device), y_test.to(self.device)

    # -------------------------------------------------------------------
    def fit(self, global_model, adapt_steps, hessian_free=True):
        self.local_model.load_state_dict(global_model.state_dict())

        for _ in range(adapt_steps):
            temp_model = copy.deepcopy(self.local_model)
            data_batch_1 = self.get_data_batch()
            grads = self.compute_grad(temp_model, data_batch_1)
            for param, grad in zip(temp_model.parameters(), grads):
                param.data.sub_(self.lr_in * grad)

            data_batch_2 = self.get_data_batch()
            grads_1st = self.compute_grad(temp_model, data_batch_2)

            data_batch_3 = self.get_data_batch()

            grads_2nd = self.compute_grad(self.local_model, data_batch_3, v=grads_1st, second_order_grads=True)
            for param, grad1, grad2 in zip(self.local_model.parameters(), grads_1st, grads_2nd):
                param.data.sub_(self.lr_out * grad1 - self.lr_out * self.lr_in * grad2)

            x_ev, y_ev = self.get_data_batch()
            y_pred = self.local_model(x_ev)
            loss = self.loss_fn(y_pred, y_ev)

        return serialize_model_params(self.local_model), loss.item()

    # -------------------------------------------------------------------
    def compute_grad(self, model, data_batch, v=None, second_order_grads=False):
        x, y = data_batch
        if second_order_grads:
            frz_model_params = copy.deepcopy(model.state_dict())
            delta = 1e-3
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for (layer_name, param), grad in zip(model.named_parameters(), v):
                    dummy_model_params_1.update({layer_name: param + delta * grad})
                    dummy_model_params_2.update({layer_name: param - delta * grad})

            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = model(x)
            loss_1 = self.loss_fn(logit_1, y)
            grads_1 = torch.autograd.grad(loss_1, model.parameters())

            model.load_state_dict(dummy_model_params_2, strict=False)
            logit_2 = model(x)
            loss_2 = self.loss_fn(logit_2, y)
            grads_2 = torch.autograd.grad(loss_2, model.parameters())

            model.load_state_dict(frz_model_params)

            grads = []
            with torch.no_grad():
                for g1, g2 in zip(grads_1, grads_2):
                    grads.append((g1 - g2) / (2 * delta))
            return grads
        else:
            logit = model(x)
            loss = self.loss_fn(logit, y)
            grads = torch.autograd.grad(loss, model.parameters())
            return grads

    # -------------------------------------------------------------------
    def perfl_eval(self, global_model, per_steps):
        # Copy the model to avoid adapting the original one
        cmodel = copy.deepcopy(global_model)

        optimizer = torch.optim.SGD(cmodel.parameters(), self.lr_ft)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        test_acc = []
        for step in range(per_steps + 1):
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


class Ditto_Client:
    def __init__(self, dataset, client_id, global_model, trainset, client_split, loss_fn, lr=0.001, mu=0.1, batch_size=20, lr_ft=0.01):
        self.trainloader, self.valloader = get_dataloader(dataset, trainset, client_split, client_id, batch_size, val_ratio=0.2)
        self.iter_trainloader = iter(self.trainloader)

        self.loss_fn = loss_fn
        self.lr = lr
        self.mu = mu
        self.lr_ft = lr_ft
        self.device = DEVICE

        self.local_model = copy.deepcopy(global_model)
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr)

        self.model_per = copy.deepcopy(self.local_model)
        self.optimizer_per = PerturbedGradientDescent(self.model_per.parameters(), lr=self.lr, mu=self.mu)

    # -------------------------------------------------------------------
    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)

        return x.to(self.device), y.to(self.device)

    # -------------------------------------------------------------------
    def get_eval_data_batch(self, size_ratio=6):
        x_test, y_test = self.get_data_batch()
        for i in range(size_ratio - 1):
            x, y = self.get_data_batch()
            x_test, y_test = torch.cat((x_test, x), dim=0), torch.cat((y_test, y), dim=0)

        return x_test.to(self.device), y_test.to(self.device)

    # -------------------------------------------------------------------
    def train(self, global_model, adapt_steps):
        self.local_model.load_state_dict(global_model.state_dict())

        for _ in range(adapt_steps):
            x, y = self.get_data_batch()
            y_pred = self.local_model(x)
            loss = self.loss_fn(y_pred, y)

            self.local_optimizer.zero_grad()
            loss.backward()
            self.local_optimizer.step()

        return serialize_model_params(self.local_model), loss.item()

    # -------------------------------------------------------------------
    def ptrain(self, adapt_steps):
        for _ in range(adapt_steps):
            x, y = self.get_data_batch()
            y_pred = self.model_per(x)
            loss = self.loss_fn(y_pred, y)

            self.optimizer_per.zero_grad()
            loss.backward()
            self.optimizer_per.step(self.local_model.parameters(), DEVICE)

        return

    # -------------------------------------------------------------------
    def fit(self, global_model, adapt_steps):
        self.ptrain(adapt_steps)
        return self.train(global_model, adapt_steps)

    # -------------------------------------------------------------------
    def perfl_eval(self, global_model, per_steps):
        # Copy the model to avoid adapting the original one
        cmodel = copy.deepcopy(global_model)

        optimizer = torch.optim.SGD(cmodel.parameters(), self.lr_ft)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        test_acc = []
        for step in range(per_steps + 1):
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


class FedRep_Client:
    def __init__(self, dataset, client_id, global_model, trainset, client_split, loss_fn, lr=0.001, batch_size=20, lr_ft=0.01):
        self.trainloader, self.valloader = get_dataloader(dataset, trainset, client_split, client_id, batch_size, val_ratio=0.2)
        self.iter_trainloader = iter(self.trainloader)

        self.loss_fn = loss_fn
        self.lr = lr
        self.lr_ft = lr_ft
        self.device = DEVICE

        self.local_model = copy.deepcopy(global_model)
        self.optimizer = torch.optim.Adam(self.local_model.base.parameters(), lr=self.lr)
        self.optimizer_per = torch.optim.Adam(self.local_model.head.parameters(), lr=self.lr)

    # -------------------------------------------------------------------
    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)

        return x.to(self.device), y.to(self.device)

    # -------------------------------------------------------------------
    def get_eval_data_batch(self, size_ratio=6):
        x_test, y_test = self.get_data_batch()
        for i in range(size_ratio - 1):
            x, y = self.get_data_batch()
            x_test, y_test = torch.cat((x_test, x), dim=0), torch.cat((y_test, y), dim=0)

        return x_test.to(self.device), y_test.to(self.device)

    # -------------------------------------------------------------------
    def fit(self, global_model, adapt_steps):
        self.local_model.load_state_dict(global_model.state_dict())

        for param in self.local_model.base.parameters():
            param.requires_grad = False
        for param in self.local_model.head.parameters():
            param.requires_grad = True

        for _ in range(adapt_steps):
            x, y = self.get_data_batch()
            y_pred = self.local_model(x)
            loss = self.loss_fn(y_pred, y)

            self.optimizer_per.zero_grad()
            loss.backward()
            self.optimizer_per.step()

        for param in self.local_model.base.parameters():
            param.requires_grad = True
        for param in self.local_model.head.parameters():
            param.requires_grad = False

        x, y = self.get_data_batch()
        y_pred = self.local_model(x)
        loss = self.loss_fn(y_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return serialize_model_params(self.local_model), loss.item()

    # -------------------------------------------------------------------
    def perfl_eval(self, global_model, per_steps):
        # Copy the model to avoid adapting the original one
        cmodel = copy.deepcopy(global_model)

        optimizer = torch.optim.SGD(cmodel.parameters(), self.lr_ft)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        test_acc = []
        for step in range(per_steps + 1):
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
