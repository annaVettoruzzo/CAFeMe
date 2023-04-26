import torch
import copy
from collections import defaultdict, OrderedDict
from fedlab.utils.dataset.sampler import SubsetSampler

from utils import DEVICE, func_call, accuracy, serialize_model_params, evaluate


class MultimodalFL_Client:
    def __init__(self, client_id, trainset, shards_part, global_model, loss_fn, lr_in, lr_out=0.001, batch_size=20):
        self.trainloader = torch.utils.data.DataLoader(trainset, sampler=SubsetSampler(indices=shards_part[client_id], shuffle=True), batch_size=batch_size)
        self.iter_trainloader = iter(self.trainloader)

        self.loss_fn = loss_fn
        self.lr_in = lr_in
        self.lr_out = lr_out
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
    def get_eval_data_batch(self, size_ratio=4):
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
    def get_adapted_parameters(self, data_batch, adapt_steps):
        phi = self.adapt(dict(self.local_model.named_parameters()), data_batch)
        for _ in range(adapt_steps - 1):
            phi = self.adapt(phi, data_batch)
        return phi

    # -------------------------------------------------------------------
    def fit(self, global_model, adapt_steps):
        self.local_model.load_state_dict(global_model.state_dict())

        data_batch_1 = self.get_data_batch()
        phi = self.get_adapted_parameters(data_batch_1, adapt_steps)

        x_2, y_2 = self.get_data_batch()
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

        optimizer = torch.optim.SGD(cmodel.parameters(), self.lr_in)

        test_acc = []
        for step in range(per_steps + 1):
            x_eval, y_eval = self.get_eval_data_batch()
            y_eval_pred = cmodel(x_eval)

            # Evaluate current model on the test data
            acc = accuracy(y_eval_pred, y_eval)
            test_acc.append(acc)

            # Adapt the model using training data
            x, y = self.get_data_batch()
            y_pred = cmodel(x)
            loss = self.loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return test_acc



class PerFedAvg_Client:
    def __init__(self, client_id, trainset, shards_part, global_model, loss_fn, lr_in, lr_out=0.001, batch_size=20):
        self.trainloader = torch.utils.data.DataLoader(trainset, sampler=SubsetSampler(indices=shards_part[client_id], shuffle=True), batch_size=batch_size)
        self.iter_trainloader = iter(self.trainloader)

        self.loss_fn = loss_fn
        self.lr_in = lr_in
        self.lr_out = lr_out
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
    def get_eval_data_batch(self, size_ratio=4):
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

        optimizer = torch.optim.SGD(cmodel.parameters(), self.lr_in)

        test_acc = []
        for step in range(per_steps + 1):
            x_eval, y_eval = self.get_eval_data_batch()
            y_eval_pred = cmodel(x_eval)

            # Evaluate current model on the test data
            acc = accuracy(y_eval_pred, y_eval)
            test_acc.append(acc)

            # Adapt the model using training data
            x, y = self.get_data_batch()
            y_pred = cmodel(x)
            loss = self.loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return test_acc
