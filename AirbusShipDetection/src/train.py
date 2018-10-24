#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import jaccard_similarity_score as jsc

# Local imports
from DefaultCNN import DefaultCNN
from ShipDataset import ShipDataset
import utils

GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0

# Images are 768x768
IMAGE_SIZE = 768
EPOCH = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.01

model = DefaultCNN()
if GPU_AVAILABLE:
    model = model.cuda()

# Define the mask for the train/validate split
data = pd.read_csv("../data/train_ship_segmentations_v2.csv")
mask = np.random.rand(len(data)) < 0.9
mask2 = np.random.rand(len(data)) >= 0.9
total_count = len(data)
train_count = (mask == True).sum()
val_count = total_count - train_count
val_count = (mask2 == False).sum()
data = None

# Prepare the dataset and the dataloader
train_data = ShipDataset(
    ship_locations_file="../data/train_ship_segmentations_v2.csv",
    img_dir="../data/train/",
    mask=mask,
    is_train=True,
    image_size=IMAGE_SIZE,
    use_cuda=GPU_AVAILABLE)
val_data = ShipDataset(
    ship_locations_file="../data/train_ship_segmentations_v2.csv",
    img_dir="../data/train/",
    mask=mask2,
    is_train=False,
    image_size=IMAGE_SIZE,
    use_cuda=GPU_AVAILABLE)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = Data.DataLoader(dataset=val_data, batch_size=int(BATCH_SIZE/4), shuffle=True)
val_iterator = iter(val_loader)

# training and testing
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = utils.iou

print("Beginning training phase..")
for epoch in range(EPOCH):
    for i, batch in enumerate(train_loader):
        output = model(batch['image'])
        loss = loss_func(output, batch['pixel_classes'])    # compute loss
        optimizer.zero_grad()                               # clear gradients for this training step
        loss.backward()                                     # backpropagation, compute gradients
        optimizer.step()                                    # apply gradients

        if (i+1) % 5 == 0:                                  # Display progress every 25 batches
            try:
                val_batch = next(val_iterator)
            except StopIteration:
                val_loader = Data.DataLoader(dataset=val_data, batch_size=int(BATCH_SIZE/4), shuffle=True)
                val_iterator = iter(val_loader)
                val_batch = next(val_iterator)
            val_output = model(val_batch['image'])
            val_loss = loss_func(val_output, val_batch['pixel_classes'])

            if GPU_AVAILABLE:
                loss = loss.cpu()
                val_loss = val_loss.cpu()

            print("Epoch: %.4f | train loss: %.4f | validation loss %.4f"
                % (epoch + float(BATCH_SIZE*(i+1) / train_count), loss.data.numpy(), val_loss.data.numpy()))
            val_loss, val_output, val_batch = None, None, None
            batch, output = None, None

print("Training complete! Saving trained model.. ", end="", flush=True)
pickle.dump(model, open('../models/model.pkl', 'wb'))
print("Done.")


""" For if I want to investigate a specific image (by index)
import matplotlib.pyplot as plt
import torchvision
to_pil = torchvision.transforms.ToPILImage()
img = to_pil(train_data[3]["image"])
plt.imshow(lum_img)
plt.title("Yoyoyoyoyo")
plt.show()
"""