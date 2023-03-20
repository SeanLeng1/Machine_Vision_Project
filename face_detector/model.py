import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3), #(kernel_size, num_filter, stride, padding
    "M", #maxpooling
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    #repeating by last item
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]



class CNNBlock(nn.Module):
    def __init__(self, in_chinnels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_chinnels, out_channels, bias =False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolo(nn.Module):
    def __init__(self, in_channels = 3, **kwargs):
        super(Yolo, self).__init__()
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(architecture_config)
        self.fc = self._create_fc(**kwargs)


    def forward(self, x):
        x = self.darknet(x)
        return self.fc(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers =[]
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                for _ in range(x[2]):
                    layers += [CNNBlock(in_channels, x[0][1], kernel_size=x[0][0], stride=x[0][2], padding = x[0][3])]
                    layers += [CNNBlock(x[0][1], x[1][1], kernel_size=x[1][0], stride=x[1][2], padding = x[1][3])]
                    in_channels = x[1][1]
        return nn.Sequential(*layers)

    def _create_fc(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(nn.Flatten(), nn.Linear(1024 * S * S, 496), nn.Dropout(0.0), nn.LeakyReLU(0.1), nn.Linear(496, S * S * (C + B * 5)))
