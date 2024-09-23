# models/lit_cnn.py

# In this file, we define a simple Convolutional Neural Network (CNN) using PyTorch Lightning.
# The LitCNN class is a subclass of pl.LightningModule and implements the forward pass, training step, validation step, test step, and optimizer configuration.
# The model architecture consists of two convolutional layers followed by ReLU activation functions and max pooling layers, and two fully connected layers.
# The forward method defines the forward pass of the network, while the training_step, validation_step, and test_step methods define the training, validation, and test steps, respectively.
# The configure_optimizers method configures the optimizer for training.


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class LitCNN(pl.LightningModule):
    """
    A LightningModule for a simple Convolutional Neural Network (CNN) using PyTorch Lightning.
    Args:
        lr (float): Learning rate for the optimizer. Default is 1e-3.
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        lr (float): Learning rate for the optimizer.
    Methods:
        forward(x):
            Defines the forward pass of the network.
        training_step(batch, batch_idx):
            Defines the training step. Computes loss and accuracy, and logs them.
        validation_step(batch, batch_idx):
            Defines the validation step. Computes loss and accuracy, and logs them.
        test_step(batch, batch_idx):
            Defines the test step. Computes loss and accuracy, and logs them.
        configure_optimizers():
            Configures the optimizer for training.
    """
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Model architecture
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # Input channels, output channels, kernel size
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)               # Kernel size, stride
        self.fc1 = nn.Linear(32 * 8 * 8, 128)        # Adjusted for image size
        self.fc2 = nn.Linear(128, 10)                # CIFAR-10 has 10 classes

        self.lr = lr # Learning rate    

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Convolutional layer followed by ReLU and pooling [Batch, 16, 16, 16]
        x = self.pool(F.relu(self.conv2(x))) # Convolutional layer followed by ReLU and pooling [Batch, 32, 8, 8]
        x = x.view(x.size(0), -1)            # Flatten the tensor [Batch, 32 * 8 * 8]
        x = F.relu(self.fc1(x))              # Fully connected layer followed by ReLU [Batch, 128]
        x = self.fc2(x)                      # Fully connected layer [Batch, 10]
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean() # Calculate accuracy
        # Log loss and accuracy
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        # Log loss and accuracy
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        # Log loss and accuracy
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]