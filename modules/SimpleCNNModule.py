import torch

# -------------------------------------------------------------------
class SimpleCNNModule(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()

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

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        x = self.flat(x)
        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = self.lin(x)
        return x
