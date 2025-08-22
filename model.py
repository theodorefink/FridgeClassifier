from torchvision import models
import torch.nn as nn
import torch.optim as optim

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # last layer of neural network for 3 options

for param in model.parameters():
    param.requires_grad = False  # freeze all layers

for param in model.fc.parameters():
    param.requires_grad = True  # unfreeze the last layer

criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # optimizer for the last layer

