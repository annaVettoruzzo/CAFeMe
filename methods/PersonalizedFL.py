import torch
import copy
from collections import defaultdict
from fedlab.utils.dataset.sampler import SubsetSampler

from utils import DEVICE, func_call, accuracy, serialize_model_params


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
        history = defaultdict(list)

        for step in range(per_steps + 1):
            x_eval, y_eval = self.get_data_batch()
            y_eval_pred = cmodel(x_eval)

            # Evaluate current model on the test data
            acc = accuracy(y_eval_pred, y_eval)
            history["pred"].append(y_eval_pred.cpu().detach())
            history["eval"].append(acc)

            # Adapt the model using training data
            x, y = self.get_data_batch()
            y_pred = cmodel(x)
            loss = self.loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return history