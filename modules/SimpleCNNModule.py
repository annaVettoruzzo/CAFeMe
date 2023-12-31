import torch


# -------------------------------------------------------------------
class CNNModule(torch.nn.Module):
    def __init__(self, conv_dim=[3, 64, 64, 64], dense_dim=[576, 576, 576], n_classes=62):
        super().__init__()

        self.cnn_block1 = self.cnn_block(conv_dim[0], conv_dim[1])
        self.cnn_block2 = self.cnn_block(conv_dim[1], conv_dim[2])
        self.cnn_block3 = self.cnn_block(conv_dim[2], conv_dim[3])
        self.flat = torch.nn.Flatten()
        self.fc = torch.nn.Sequential(
            self.dense_block(dense_dim[0], dense_dim[1]),
            self.dense_block(dense_dim[1], dense_dim[2]),
            torch.nn.Linear(dense_dim[2], n_classes)
        )

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
        x = self.fc(x)
        return x


# -------------------------------------------------------------------
class SimpleCNNModule(torch.nn.Module):
    def __init__(self, conv_dim=[1, 32, 64], dense_dim=[1024, 512], n_classes=10):
        super().__init__()

        self.cnn_block1 = self.cnn_block(conv_dim[0], conv_dim[1])
        self.cnn_block2 = self.cnn_block(conv_dim[1], conv_dim[2])
        self.flat = torch.nn.Flatten()
        self.fc = torch.nn.Sequential(
            self.dense_block(dense_dim[0], dense_dim[1]),
            torch.nn.Linear(dense_dim[1], n_classes)
        )

    def cnn_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 5),
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
        x = self.flat(x)
        x = self.fc(x)
        return x
