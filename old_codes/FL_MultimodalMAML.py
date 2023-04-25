import torch
import copy
from collections import defaultdict

from data import get_dataloader
from utils import func_call, DEVICE, local_evaluation
from fedlab.utils.serialization import SerializationTool


class FL_Client:
    def __init__(self, client_id, dataset, global_model, loss_fn, lr_in, lr_out=0.001, adapt_steps=2, batch_size=20, valset_ratio=0.1):
        self.trainloader, self.valloader = get_dataloader(dataset, client_id, batch_size, valset_ratio)
        self.iter_trainloader = iter(self.trainloader)

        self.local_model = copy.deepcopy(global_model)
        self.loss_fn = loss_fn
        self.lr_in = lr_in
        self.lr_out = lr_out
        self.adapt_steps = adapt_steps
        self.device = DEVICE

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
    """
    Takes a support set (X, y) corresponding to a specific task, and returns the task specific 
    parameters phi (after adapting theta with GD using one or multiple adaptation steps)
    """

    def get_adapted_parameters(self, data_batch):
        phi = self.adapt(dict(self.local_model.named_parameters()), data_batch)
        for _ in range(self.adapt_steps - 1):
            phi = self.adapt(phi, data_batch)
        return phi

    # -------------------------------------------------------------------
    def fit(self, global_model, local_steps):

        self.local_model.load_state_dict(global_model.state_dict())
        local_optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr_out)

        tot_loss = 0.0
        for _ in range(local_steps):
            # Personalize (adapt)
            data_batch_sp = self.get_data_batch()
            phi = self.get_adapted_parameters(data_batch_sp)

            # Evaluate
            x_qr, y_qr = self.get_data_batch()
            y_qr_pred = func_call(self.local_model, phi, x_qr)
            tot_loss += self.loss_fn(y_qr_pred, y_qr)

        tot_loss = tot_loss / local_steps

        local_optimizer.zero_grad()
        tot_loss.backward()
        local_optimizer.step()

        return SerializationTool.serialize_model(self.local_model)

    # -------------------------------------------------------------------
    def pfl_eval(self, global_model, steps):
        self.local_model.load_state_dict(global_model.state_dict())
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.lr_in)
        history = defaultdict(list)

        history["loss_before"], history["acc_before"] = local_evaluation(self.local_model, self.valloader, self.loss_fn)

        for _ in range(steps+1):
            x, y = self.get_data_batch()
            y_pred = func_call(self.local_model, None, x)
            loss = self.loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        history["loss_after"], history["acc_after"] = local_evaluation(self.local_model, self.valloader, self.loss_fn)

        return history


