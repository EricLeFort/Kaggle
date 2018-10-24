#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import torch
import torch.nn as nn

#The size of the incoming images are 3(colours) x 768(width) x 768(height)
class DefaultCNN(nn.Module):
    """
    The architecture of the Convolutional Neural Network being used
    """
    def __init__(self):
        super(DefaultCNN, self).__init__()             # input shape (3, 768, 768)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=6,
                stride=2,
                padding=3
            ),                                  # output shape (6, 384, 384)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=4,
                stride=2
            )                                   # output shape (6, 191, 191)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=1
            ),                                  # output shape (18, 191, 191)
            nn.Conv2d(
                in_channels=18,
                out_channels=36,
                kernel_size=3,
                stride=2,
                padding=0
            ),                                  # output shape (36, 95, 95)
            nn.Conv2d(
                in_channels=36,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=1
            ),                                  # output shape (18, 95, 95)
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=18,
                out_channels=6,
                kernel_size=3,
                stride=2,
                padding=1
            ),                                  # output shape (6, 48, 48)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=6,
                stride=6
            )                                   # output shape (6, 8, 8)
        )

        self.out = nn.Sequential(
            #nn.Dropout(p=0.5, inplace=True),
            nn.Linear(
                int(6 * 8 * 8),
                589824
            ),
            nn.Sigmoid()                        # output shape (589824)
        )

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.view(x.size(0), -1)               # flatten to (batch_size, 32*image_size/4*image_size/4)
        y = self.out(x)
        return y.view(-1, 768, 768)             # Final output: batch_size, 768, 768