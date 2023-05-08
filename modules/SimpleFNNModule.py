import torch


# -------------------------------------------------------------------
class SimpleFNNModule(torch.nn.Module):
    def __init__(self, dense_dim=[784, 200, 200], n_classes=10):
        super().__init__()

        self.flat = torch.nn.Flatten()
        self.dense_block1 = self.dense_block(dense_dim[0], dense_dim[1])
        self.dense_block2 = self.dense_block(dense_dim[1], dense_dim[2])
        self.lin = torch.nn.Linear(dense_dim[2], n_classes)


    def dense_block(self, dim_in, dim_out):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.BatchNorm1d(dim_out, track_running_stats=False),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.flat(x)
        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = self.lin(x)
        return x

