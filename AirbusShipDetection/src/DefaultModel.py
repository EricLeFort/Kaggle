#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import torch.nn as nn

#The size of the incoming images are 3(colours) x 768(width) x 768(height)
class CNN(nn.Module):
    """
    The architecture of the Convolutional Neural Network being used
    """
    def __init__(self):
        super(CNN, self).__init__()             # input shape (3, 768, 768)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,                  # input height (3 colours)
                out_channels=15,                # 3 colours * 5 filters
                kernel_size=7,                  # size of the convolution
                stride=3,                       # stride of convolution
                padding=2,                      # size of 0-padding
            ),                                  # output shape (15, 256, 256)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=4,
                stride=4
            )                                   # output shape (15, 64, 64)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=15,
                out_channels=30,
                kernel_size=2,
                stride=2,
                padding=0
            ),                                  # output shape (60, 32, 32)
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=30,
                out_channels=8,                # Shrink from 60 channels down to 8
                kernel_size=2,
                stride=2,
                padding=0
            ),                                  # output shape (8, 16, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=4,
                stride=4
            )                                   # output shape (8, 4, 4)
        )

        self.out = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(
                int(8 * 4 * 4),
                589824
            ),
            nn.Softmax(1)                       # output shape (1, 589824)
        )

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.view(x.size(0), -1)               # flatten to (batch_size, 32*image_size/4*image_size/4)
        output = self.out(x)
        return output.view(-1, 768, 768)        # Final output: batch_size, 768, 768