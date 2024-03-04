import torch
import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms

from torchvision.models import resnet50, ResNet50_Weights

torch.set_num_threads(2)
model_weights = ResNet50_Weights.IMAGENET1K_V1

model = resnet50(weights = model_weights)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 512),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(512, 3),
    torch.nn.Sigmoid()
)
model.load_state_dict(torch.load('bestmodel.pth'))

transform = model_weights.transforms()

img = Image.open('461.png')
img = transforms.ToTensor()(img)
img = img.repeat(3, 1, 1)
#img = self.augmentation(img)
input = transform(img)

input = input.unsqueeze(0)

model.eval()
output = model(input)
threshold = 0.5
predicted_probs = output.detach().numpy()  
predicted_classes = (predicted_probs > threshold).astype(int)  
sehat = [1, 0, 0]
nonProliferatik = [0, 1, 0]
Proliferatik = [0, 0, 1]

if (predicted_classes == sehat).all():
    hasil = "SEHAT"
elif (predicted_classes == nonProliferatik).all():
    hasil = "Diabetes Non-proliferatik"
elif (predicted_classes == Proliferatik).all():
    hasil = "Diabetes Proliferatik"
    
print(output)