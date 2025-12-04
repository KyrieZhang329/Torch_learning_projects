import torch
import torch.nn as nn


class ObjectRecognitionNet(nn.Module):
    def __init__(self):
        super(ObjectRecognitionNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, 
                     stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, 
                     stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, 
                     stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x