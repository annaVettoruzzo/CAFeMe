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

        self.cnn_block1 = self.cnn_block(3, 64)
        self.cnn_block2 = self.cnn_block(64, 64)
        self.cnn_block3 = self.cnn_block(64, 64)
        self.flat = torch.nn.Flatten()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(1024, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            LambdaLayer(lambda x: torch.mean(x, dim=0)),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, out_dim),
            LambdaLayer(lambda x: reshape(x))
        )

    def cnn_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            torch.nn.BatchNorm2d(out_channels, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        x = self.flat(x)
        x = self.net(x)
        return x


# -------------------------------------------------------------------
class SimpleCNNModuleWithTE(torch.nn.Module):
    def __init__(self, n_classes, modulation):
        super().__init__()

        self.modulation = modulation
        if modulation in ["c0", "c1"]:   self.te = TaskEncoderComplex(out_shapes=[(1, 64, 1, 1), (1, 64, 1, 1), (1, 64, 1, 1), (1, 1024), (1, 576), (1, 576)])
        elif modulation in ["c2"]:       self.te = TaskEncoderComplex(out_shapes=[(2, 64, 1, 1), (2, 64, 1, 1), (2, 64, 1, 1), (2, 1024), (2, 576), (2, 576)])

        self.cnn_block1 = self.cnn_block(3, 64)
        self.cnn_block2 = self.cnn_block(64, 64)
        self.cnn_block3 = self.cnn_block(64, 64)
        self.flat = torch.nn.Flatten()
        self.dense_block1 = self.dense_block(1024, 576)
        self.dense_block2 = self.dense_block(576, 576)
        self.lin = torch.nn.Linear(576, n_classes)

    def cnn_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            torch.nn.BatchNorm2d(out_channels, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )

    def dense_block(self, dim_in, dim_out):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.BatchNorm1d(dim_out, track_running_stats=False),
            torch.nn.ReLU(),
        )

    def modulate(self, x, z):
        if   self.modulation in ["c0"]: return x + z
        elif self.modulation in ["c1"]: return x * torch.sigmoid(z)
        elif self.modulation in ["c2"]: return x * z[0] + z[1]

    def forward(self, x):
        r1, r2, r3, z0, z1, z2 = self.te(x)

        x = self.cnn_block1(x)
        x = self.modulate(x, r1)

        x = self.cnn_block2(x)
        x = self.modulate(x, r2)

        x = self.cnn_block3(x)
        x = self.modulate(x, r3)

        x = self.flat(x)
        x = self.modulate(x, z0)

        x = self.dense_block1(x)
        x = self.modulate(x, z1)

        x = self.dense_block2(x)
        x = self.modulate(x, z2)

        x = self.lin(x)
        return x

