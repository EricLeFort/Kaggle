#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

# Local imports
from DefaultModel import CNN
from ShipDataset import ShipDataset
import utils

# Images are 768x768
IMAGE_SIZE = 768
EPOCH = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def bounding_box_to_coordinate_runs(x, y, width, height):
    runs = []
    row_start = (y-1) * IMAGE_SIZE
    for i in range(0, height):
        runs.append((row_start + x, width))
        row_start += IMAGE_SIZE

    return runs

model = CNN()

# Define the mask for the train/validate split
data = pd.read_csv("../data/train_ship_segmentations_v2.csv")
mask = np.random.rand(len(data)) < 0.9
total_count = len(data)
train_count = (mask == True).sum()
val_count = total_count - train_count
data = None

# Prepare the dataset and the dataloader
train_data = ShipDataset(
    ship_locations_file="../data/train_ship_segmentations_v2.csv",
    img_dir="../data/train/",
    mask=mask,
    is_train=True,
    image_size=IMAGE_SIZE)
val_data = ShipDataset(
    ship_locations_file="../data/train_ship_segmentations_v2.csv",
    img_dir="../data/train/",
    mask=mask,
    is_train=False,
    image_size=IMAGE_SIZE)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE*4, shuffle=True)

# training and testing
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.BCELoss()
print("Beginning training phase..")
for epoch in range(EPOCH):
    for i, batch in enumerate(train_loader):
        output = model(batch['image'])
        loss = loss_func(output, batch['pixel_classes'])    # compute loss
        optimizer.zero_grad()                               # clear gradients for this training step
        loss.backward()                                     # backpropagation, compute gradients
        optimizer.step()                                    # apply gradients

        if (i+1) % 5 == 0:                                  # Display progress every 5 batches
            val_batch = next(iter(val_loader))
            val_output = model(val_batch['image'])
            val_loss = loss_func(val_output, val_batch['pixel_classes'])
            print("Epoch: %.4f | train loss: %.4f | validation loss %.4f"
                % (float(BATCH_SIZE*(i+1) / train_count), loss.data.numpy(), val_loss.data.numpy()))




""" For if I want to investigate a specific image (by index)
import matplotlib.pyplot as plt
import numpy as np
to_pil = torchvision.transforms.ToPILImage()
img = to_pil(train_data[3]["image"])
plt.imshow(lum_img)
plt.title("Yoyoyoyoyo")
plt.show()
"""