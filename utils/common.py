import torch
import os
import numpy as np
from .stateless import functional_call
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------
os.system('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free >tmp')
memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
gpu_number = int(np.argmax(memory_available))
torch.cuda.set_device(gpu_number)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

IMAGE_SIZE = 32

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