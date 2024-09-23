# utils/datamodule.py

# In this file, we define a PyTorch Lightning DataModule for the CIFAR-10 dataset.
# The CIFAR10DataModule class is a subclass of pl.LightningDataModule and implements the prepare_data, setup, train_dataloader, val_dataloader, and test_dataloader methods.
# The prepare_data method downloads the CIFAR-10 dataset if it is not already present in the data directory.
# The setup method sets up the CIFAR-10 dataset for training and testing.
# The train_dataloader, val_dataloader, and test_dataloader methods return DataLoaders for the training, validation, and test datasets, respectively.

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

class CIFAR10DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the CIFAR-10 dataset.
    Args:
        batch_size (int): The batch size to use for the dataloaders. Default is 64.
        data_dir (str): The directory where the CIFAR-10 data will be stored. Default is 'data'.
    Attributes:
        batch_size (int): The batch size to use for the dataloaders.
        data_dir (str): The directory where the CIFAR-10 data will be stored.
        transform (torchvision.transforms.Compose): The transformations to apply to the data.
    Methods:
        prepare_data():
            Downloads the CIFAR-10 dataset if it is not already present in the data directory.
        setup(stage=None):
            Sets up the CIFAR-10 dataset for training and testing.
        train_dataloader():
            Returns a DataLoader for the training dataset.
        val_dataloader():
            Returns a DataLoader for the validation dataset.
        test_dataloader():
            Returns a DataLoader for the test dataset.
    """
    def __init__(self, batch_size=64, data_dir='data'):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

        self.train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

        self.test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.cifar10_train = datasets.CIFAR10(self.data_dir, train=True, transform=self.train_transform)
        self.cifar10_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False)
    