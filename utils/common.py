import torch
import os
import numpy as np
import pickle
from tfrecord.torch.dataset import TFRecordDataset
from cv2 import imdecode
from .stateless import functional_call
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------
os.system('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free >tmp')
memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
gpu_number = int(np.argmax(memory_available))
torch.cuda.set_device(gpu_number)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# -------------------------------------------------------------------
def load_tfrecord_images(fpath):
    dataset = TFRecordDataset(fpath, None, {"image": "byte", "label": "int"})
    dataset = list(dataset)
    label = dataset[0]["label"][0]
    images = [imdecode(dico["image"], -1) for dico in dataset]
    return images, label


# -------------------------------------------------------------------
def find_indices(list_to_check, item_to_find):
    array = np.array(list_to_check)
    indices = np.where(array == item_to_find)[0]
    return list(indices)


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
def deserialize_specific_model_params(model, params_list, name):
    """ Copy parameters that contains name in params_list inside model """
    current_index = 0  # keep track of where to read from grad_update
    for param_name, parameter in model.named_parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        if name in param_name:
            parameter.data.copy_(params_list[current_index:current_index + numel].view(size))
        current_index += numel
    return


# -------------------------------------------------------------------
def aggregate_model_params(params_list):
    """ As in FedAVg considering the same weight for all the clients """
    aggregate_parameters = torch.mean(torch.stack(params_list, dim=-1), dim=-1)
    return aggregate_parameters


# -------------------------------------------------------------------
def evaluate_client(model, dataloader):
    tot_acc = []
    for x, y in dataloader:
        if len(y) == 1: continue
        x, y = x.to(DEVICE), y.to(DEVICE)
        logit = model(x)
        tot_acc.append(accuracy(logit, y))
    return np.mean(tot_acc)


# -------------------------------------------------------------------
def evaluate_fl(global_model, clients, clients_test, steps=100, fine_tuning=True, save=""):
    tot_acc = []
    for client_id in clients_test:
        if fine_tuning:
            eval_acc = clients[client_id].perfl_eval(global_model, steps)
        else:
            eval_acc = clients[client_id].fl_eval(global_model)
        tot_acc.append(eval_acc)

    avg_acc = np.mean(tot_acc, axis=0)

    if save:
        with open(f'{save}.pkl', 'wb') as file:
            pickle.dump(avg_acc, file)
    return avg_acc


