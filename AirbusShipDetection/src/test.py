#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from skimage import io

root_path = "../data/test/"
files = os.listdir(root_path)

# Load the model
model = pickle.load(open("../models/model1.pkl", 'rb'))

all_runs = []
for file in files:
    print(file)

    # Read in image, convert from numpy image to torch image
    # numpy image: H x W x C
    # torch image: C X H X W
    image = io.imread(root_path + file)
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    image = image.type(torch.FloatTensor)

    # Make the prediction and flatten it
    pred = model(image.unsqueeze(0))
    pred = pred.view(-1)

    # Clamp to the range (0, 1), round to the nearest whole number
    pred = torch.clamp(pred, 0, 1)
    pred = torch.round(pred)
    pred = pred.type(torch.ByteTensor)
    pred = pred.numpy()

    # Convert predictions to runs
    runs = ""
    in_run = False
    start = -1
    for pixel in range(0, len(pred)):
        if in_run:
            if pred[pixel] == 0:
                in_run = False
                runs += str(start) + " " + str(pixel - start + 1) + " "
        else:
            if pred[pixel] == 1:
                in_run = True
                start = pixel + 1

    if in_run:
        runs += str(start) + " " + str(len(pred) - start + 1) + " "
    all_runs.append([file, runs[:-1]])

all_runs = pd.DataFrame(all_runs, columns=["ImageId", "EncodedPixels"])
all_runs.to_csv("../predictions/prediction1.csv", index=False)


# TODO iterate through each of these files
#   Append this prediction to the predictions
#   predictions.to_csv("../predictions/model1.csv")