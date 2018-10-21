import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Local imports
import utils

class ShipDataset(Dataset):
    """Ship Detection Dataset"""

    def __init__(self, ship_locations_file, img_dir, mask, is_train, image_size, use_cuda=False, transform=None):
        """
        Initializes the Ship Detection Dataset
        Args:
            ship_locations_file (string): The path to the file containing the annotated ship locations
            img_dir (string): The path to the directory containing the images
            mask ():
            train (bool): Whether to use the training set or the validation set
            image_size (int): The size of the image (it is square for this dataset)
            transform (callable, optional): Optional transform to be applied to a sample
        """
        self.ship_locations = pd.read_csv(
            ship_locations_file,
            dtype={
                "ImageId": object,
                "EncodedPixels": object
            })
        self.ship_locations = self.ship_locations.fillna('')
        self.img_dir = img_dir
        self.image_size = image_size
        self.transform = transform
        self.use_cuda = use_cuda

        if is_train:
            self.ship_locations = self.ship_locations[mask]
        else:
            self.ship_locations = self.ship_locations[~mask]
    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return self.ship_locations.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the item at the index "idx" in the dataset
        Args:
            idx (int): The index of the item to retrieve
        """
        path = self.ship_locations.iloc[idx, 0]
        image = io.imread(self.img_dir + path)

        if self.ship_locations.iloc[idx, 1] == "":
            pixel_classes = torch.zeros([self.image_size, self.image_size], dtype=torch.float)
        else:
            # Convert runs from a string to be a list of tuples
            runs = self.ship_locations.iloc[idx, 1].split(' ')
            runs = list(zip(runs[::2], runs[1::2]))
            runs = list(map(lambda run: (float(run[0]), float(run[1])), runs))

            # Convert the list of tuples into a class for each pixel
            pixel_classes = utils.runs_to_pixel_classes(runs, self.image_size)

        # Convert from numpy image to torch image
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)

        if self.use_cuda:
            image = image.cuda()
            pixel_classes = pixel_classes.cuda()

        return {'image': image, 'pixel_classes': pixel_classes}