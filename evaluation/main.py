import torch
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

torch.set_num_threads(2)
model_weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1

model = mobilenet_v3_small(weights = model_weights)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(576, 258),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(258, 3),
    torch.nn.Sigmoid()
)
model.load_state_dict(torch.load("./evaluation/bestmodel.pth"))

transform = model_weights.transforms()

img = Image.open('./evaluation/461.png')
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
    
print(hasil)