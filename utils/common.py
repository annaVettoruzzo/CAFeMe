import torch
import os
import numpy as np
from .stateless import functional_call
from sklearn.metrics import accuracy_score

IMAGE_SIZE = 32

# -------------------------------------------------------------------
os.system('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free >tmp')
memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
gpu_number = int(np.argmax(memory_available))
torch.cuda.set_device(gpu_number)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# -------------------------------------------------------------------
class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# -------------------------------------------------------------------
def func_call(model, params_dict, x):
    if params_dict is None: params_dict = dict(model.named_parameters())
    y = functional_call(model, params_dict, x)
    return y


# -------------------------------------------------------------------
def accuracy(pred, y_true):
    y_pred = pred.argmax(1).reshape(-1).cpu()
    y_true = y_true.reshape(-1).cpu()
    return accuracy_score(y_pred, y_true)


# -------------------------------------------------------------------
def serialize_model_params(model):
    """ Serialize model parameters in a single tensor """
    parameters = [param.data.view(-1) for param in model.parameters()]
    m_parameters = torch.cat(parameters)
    m_parameters = m_parameters.cpu()
    return m_parameters


# -------------------------------------------------------------------
def deserialize_model_params(model, params_list):
    """ Copy parameters in params_list inside model """
    current_index = 0  # keep track of where to read from grad_update
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        parameter.data.copy_(params_list[current_index:current_index + numel].view(size))
        current_index += numel
    return


# -------------------------------------------------------------------
def aggregate_model_params(params_list):
    """ As in FedAVg considering the same weight for all the clients """
    aggregate_parameters = torch.mean(torch.stack(params_list, dim=-1), dim=-1)
    return aggregate_parameters
