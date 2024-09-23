# models/__init__.py

from .lit_cnn import LitCNN
from .resnet import LitResNet

# Dictionary mapping model names to classes
model_dict = {
    'LitCNN': LitCNN,
    'ResNet': LitResNet,
}
