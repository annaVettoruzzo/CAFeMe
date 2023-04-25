import torch
import numpy as np
from utils.common import LambdaLayer


# -------------------------------------------------------------------
class TaskEncoderComplex(torch.nn.Module):
    def __init__(self, out_shapes):
        super().__init__()
        out_dims = [np.prod(shape) for shape in out_shapes]
        out_dim = np.sum(out_dims)

        def reshape(x):
            lst = x.split(out_dims)
            return [z.view(shape) for z, shape in zip(lst, out_shapes)]

        self.fc1 = torch.nn.Linear(28 * 28, 80)
        self.fc2 = torch.nn.Linear(80, 60)
        self.activation = torch.nn.ReLU()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(60, 25),
            torch.nn.ReLU(),
            LambdaLayer(lambda x: torch.mean(x, dim=0)),
            torch.nn.Linear(25, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, out_dim),
            LambdaLayer(lambda x: reshape(x))
        )

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.net(x)
        return x


# -------------------------------------------------------------------
class SimpleLinModuleWithTE(torch.nn.Module):
    def __init__(self, n_classes, modulation):
        super().__init__()

        self.modulation = modulation
        if modulation in ["c0", "c1"]:
            self.te = TaskEncoderComplex(out_shapes=[(1, 80), (1, 60)])
        elif modulation in ["c2"]:
            self.te = TaskEncoderComplex(out_shapes=[(2, 80), (2, 60)])

        self.fc1 = torch.nn.Linear(28 * 28, 80)
        self.fc2 = torch.nn.Linear(80, 60)
        self.fc3 = torch.nn.Linear(60, n_classes)
        self.flat = torch.nn.Flatten()
        self.activation = torch.nn.ReLU()

    def modulate(self, x, z):
        if self.modulation in ["c0"]:
            return x + z
        elif self.modulation in ["c1"]:
            return x * torch.sigmoid(z)
        elif self.modulation in ["c2"]:
            return x * z[0] + z[1]

    def forward(self, x):
        x = self.flat(x)
        z1, z2 = self.te(x)

        x = self.activation(self.fc1(x))
        x = self.modulate(x, z1)

        x = self.activation(self.fc2(x))
        x = self.modulate(x, z2)

        x = self.fc3(x)
        return x

