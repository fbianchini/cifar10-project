# scripts/train.py

# In this script, we define a function train_model that trains a simple Convolutional Neural Network (CNN) using PyTorch Lightning.
# The model is trained on the CIFAR-10 dataset using the CIFAR10DataModule class defined in utils/datamodule.py.
# The trained model is saved to a checkpoint file named models/lit_cnn.ckpt.

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from models import model_dict
from utils.datamodule import CIFAR10DataModule

def train_model(args):
    pl.seed_everything(42)  # Set a global random seed for reproducibility

    # Get the model class from the model dictionary
    if args.model_name in model_dict:
        ModelClass = model_dict[args.model_name]
    else:
        raise ValueError(f"Model {args.model_name} not found in model_dict")

    # Create a Lightning model
    model = ModelClass(lr=args.learning_rate)

    # Create a Lightning DataModule
    data_module = CIFAR10DataModule(batch_size=args.batch_size)
    
    # Define a TensorBoard logger
    logger = TensorBoardLogger('logs/', name=args.experiment_name)

    # Define callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='models/',
        filename=f'{args.experiment_name}' + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    # Create a Trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1  # Set devices=1 for both GPU and CPU
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)
    print('Training complete. Best model saved at:', checkpoint_callback.best_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN on CIFAR-10')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to train')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='size of each training batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for optimizer')
    parser.add_argument('--experiment_name', type=str, default='cifar10_experiment', help='name of the experiment')

    args = parser.parse_args()
    train_model(args)
