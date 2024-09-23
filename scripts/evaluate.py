# scripts/evaluate.py

# In this script, we define a function evaluate_model that loads the trained model from the checkpoint file models/lit_cnn.ckpt, 
# creates a Lightning DataModule using the CIFAR10DataModule class, and then tests the model using the test dataset. 
# The test accuracy is printed to the console.

import argparse
import torch
import pytorch_lightning as pl
from models import model_dict
from utils.datamodule import CIFAR10DataModule

def evaluate_model(args):

    # Get the model class from the model dictionary
    if args.model_name in model_dict:
        ModelClass = model_dict[args.model_name]
    else:
        raise ValueError(f"Model {args.model_name} not found in model_dict")

    # Load the trained model
    model = ModelClass.load_from_checkpoint(args.checkpoint_path)
    
    # Create a Lightning DataModule
    data_module = CIFAR10DataModule(batch_size=args.batch_size)
    
    # Create a Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1  # Use 1 GPU or 1 CPU based on availability
    )
    
    # Test the model
    result = trainer.test(model, datamodule=data_module)
    print(f"Test Accuracy: {result[0]['test_acc'] * 100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained CNN on CIFAR-10')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to evaluate')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for evaluation')

    args = parser.parse_args()
    evaluate_model(args)


