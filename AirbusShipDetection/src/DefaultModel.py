#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import torch.nn as nn


class CNN(nn.Module):
    """
    The architecture of the Convolutional Neural Network being used
    """
    def __init__(self, image_size):
        super(CNN, self).__init__()
        self.KERNEL_SIZE = 5
        self.STRIDE = 1
        self.PADDING = 2

        self.conv1 = nn.Sequential(             # input shape (1, image_size, image_size)
            nn.Conv2d(
                in_channels=3,                  # input height
                out_channels=16,                # n_filters
                kernel_size=self.KERNEL_SIZE,   # filter size
                stride=self.STRIDE,             # filter movement/step
                padding=self.PADDING,           # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                                  # output shape (16, image_size, image_size)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),        # choose max value in 2x2 area, output shape (16, image_size/2, image_size/2)
        )

        self.conv2 = nn.Sequential(             # input shape (16, image_size/2, image_size/2)
            nn.Conv2d(16, 32, 5, 1, 2),         # output shape (32, image_size/2, image_size/2)
            nn.ReLU(),
            nn.MaxPool2d(2),                    # output shape (32, image_size/4, image_size/4)
        )

        self.out = nn.Linear(int(32 * image_size/4 * image_size/4), 589824) # fully connected layer, output: 768x768x2

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)               # flatten to (batch_size, 32*image_size/4*image_size/4)
        output = self.out(x)
        return output.view(self.image_size, -1)