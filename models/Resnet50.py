import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet50, self).__init__()

        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def resnet50(pretrained=False, num_classes=1000):
    model = ResNet50(num_classes=num_classes, pretrained=pretrained)
    return model
