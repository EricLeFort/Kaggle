#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import torch
import torch.nn as nn

#The size of the incoming images are 3(colours) x 768(width) x 768(height)
class PureCNN(nn.Module):
    """
    The architecture of the Convolutional Neural Network being used
    """
    def __init__(self):
        super(PureCNN, self).__init__()			# input shape (3, 768, 768)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=11,
                stride=1,
                padding=5
            ),                                  # output shape (6, 768, 768)
            nn.Conv2d(
                in_channels=6,
                out_channels=12,
                kernel_size=7,
                stride=1,
                padding=3
            ),                                  # output shape (12, 768, 768)
            nn.Conv2d(
                in_channels=12,
                out_channels=12,
                kernel_size=5,
                stride=1,
                padding=2
            )                                   # output shape (12, 768, 768)
        )

        self.out = nn.Sequential(
        	nn.Conv2d(
                in_channels=12,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Dropout(p=0.5, inplace=True),
            nn.Sigmoid()                        # output shape (768, 768)
        )

    def forward(self, x):
        x = self.conv(x)
        y = self.out(x)
        return y.view(-1, 768, 768)             # Final output: batch_size, 768, 768