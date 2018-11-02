#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import torch
import torch.nn as nn

def build_encode_block(in_channels: int, out_channels: int, num_layers: int):
    block = nn.Sequential()

    # First layer: increase channel size from in_channels to out_channels
    block.add_module(
        "conv0",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
    )
    block.add_module("norm0", nn.BatchNorm2d(out_channels))
    block.add_module("relu0", nn.ReLU(inplace=True))

    # Add the remaining layers, 3x3 convolutions that maintain the same dimensions and channel size
    for i in range(1, num_layers):
        block.add_module(
            "conv" + str(i),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        block.add_module("norm" + str(i), nn.BatchNorm2d(out_channels))
        block.add_module("relu" + str(i), nn.ReLU(inplace=True))

    # Add the 2x2 pooling with stride 2, this cuts the dimensions in half
    block.add_module(
        "pool",
        nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
    )

    return block

def build_decode_block(in_channels: int, out_channels: int, num_layers: int):
    block = nn.Sequential()

    # First layer: deconvolutional layer, upsamples to double the dimensions
    block.add_module(
        "deconv",
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
    )

    # Add all but the last convolutional layers, 3x3 convolutions that maintain the same dimensions and channel size
    for i in range(1, num_layers-1):
        block.add_module(
            "conv" + str(i),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        block.add_module("norm" + str(i), nn.BatchNorm2d(in_channels))
        block.add_module("relu" + str(i), nn.ReLU(inplace=True))

    # Last layer: decrease channel size from in_channels to out_channels
    block.add_module(
        "conv" + str(num_layers),
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
    )
    block.add_module("norm" + str(num_layers), nn.BatchNorm2d(out_channels))
    block.add_module("relu" + str(num_layers), nn.ReLU(inplace=True))

    return block

#The size of the incoming images are 3(colours) x 768(width) x 768(height)
class EncoderDecoder(nn.Module):
    """
    The architecture of the Convolutional Neural Network being used
    """
    def __init__(self):
        super(EncoderDecoder, self).__init__()  # input shape (3, 768, 768)

        self.encode1 = build_encode_block(
            in_channels=3,
            out_channels=8,
            num_layers=1
        )                                       # output shape (3, 384, 384)
        self.encode2 = build_encode_block(
            in_channels=8,
            out_channels=8,
            num_layers=2
        )                                       # output shape (8, 192, 192)
        self.encode3 = build_encode_block(
            in_channels=8,
            out_channels=16,
            num_layers=3
        )                                       # output shape (16, 96, 96)
        self.encode4 = build_encode_block(
            in_channels=16,
            out_channels=32,
            num_layers=3
        )                                       # output shape (32, 48, 48)
        self.encode5 = build_encode_block(
            in_channels=32,
            out_channels=64,
            num_layers=3
        )                                       # output shape (64, 24, 24)

        self.decode1 = build_decode_block(
            in_channels=64,
            out_channels=32,
            num_layers=3
        )                                       # output shape (32, 48, 48)
        self.decode2 = build_decode_block(
            in_channels=32,
            out_channels=16,
            num_layers=3
        )                                       # output shape (16, 96, 96)
        self.decode3 = build_decode_block(
            in_channels=16,
            out_channels=8,
            num_layers=3
        )                                       # output shape (8, 192, 192)
        self.decode4 = build_decode_block(
            in_channels=8,
            out_channels=3,
            num_layers=2
        )                                       # output shape (3, 384, 384)
        self.decode5 = build_decode_block(
            in_channels=3,
            out_channels=1,
            num_layers=1
        )                                       # output shape (3, 768, 768)

        self.out = nn.Sequential(
            nn.ReLU()                           # output shape (768, 768)
        )

    def forward(self, x):
        encode = self.encode1(x)
        encode = self.encode2(encode)
        encode = self.encode3(encode)
        encode = self.encode4(encode)
        encode = self.encode5(encode)

        decode = self.decode1(encode)
        decode = self.decode2(decode)
        decode = self.decode3(decode)
        decode = self.decode4(decode)
        decode = self.decode5(decode)

        y = self.out(decode)
        y = y.clamp(0, 1)
        return y.squeeze()                      # Final output: batch_size, 768, 768