import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim

import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
import torch.nn as nn

# Dataloading for Diabetic Retinopathy Image Classification

if __name__ == '__main__' :

    path = "Dataset/FullDataset/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv"
    df = pd.read_csv(path)

    class DiabeticRetinopathyDataset(Dataset):
        def __init__(self, label_df, img_dir, transform=None):
            self.img_labels = label_df
            self.img_dir = img_dir
            self.transform = transform
        
        def __len__(self):
            return len(self.img_labels)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = Image.open(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            return image, label

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),             # Random rotation within 30 degrees
        transforms.RandomHorizontalFlip(),         # Random horizontal flip
        transforms.RandomVerticalFlip(),           # Random vertical flip
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming single-channel grayscale images
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming single-channel grayscale images
    ])

    # Split the data into training and validation sets while maintaining class balance
    train_data = []
    val_data = []
    for class_label in df['DR grade'].unique():
        class_data = df[df['DR grade'] == class_label]
        train_group, val_group = train_test_split(class_data, test_size=0.2, random_state=42)
        train_data.append(train_group)
        val_data.append(val_group)

    train_df = pd.concat(train_data)
    val_df = pd.concat(val_data)

    # Count the number of images per class
    class_counts = train_df['DR grade'].value_counts()

    # Print the counts for each class
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} images")

    # Count the number of images per class
    class_counts = val_df['DR grade'].value_counts()

    # Print the counts for each class
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} images")

    img_dir = "Dataset/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set"
    training_data = DiabeticRetinopathyDataset(train_df, img_dir, train_transform)
    validation_data = DiabeticRetinopathyDataset(val_df, img_dir, val_transform)

    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=32, shuffle=False)

    class DiabeticRetinopathyModel(nn.Module):
        def __init__(self, num_classes=3):
            super(DiabeticRetinopathyModel, self).__init__()
            # Load pre-trained ResNet-50 model
            self.resnet_50 = torch.hub.load("pytorch/vision", "resnet152", weights="IMAGENET1K_V2")

            # Freeze the parameters of the pre-trained layers
            for param in self.resnet_50.parameters():
                param.requires_grad = False

            # Modify the last fully connected layer to have num_classes output classes
            num_ftrs = self.resnet_50.fc.in_features
            self.resnet_50.fc = nn.Linear(num_ftrs, num_classes)

        def forward(self, x):
            return self.resnet_50(x)

    # Example usage:
    # Instantiate the model
    model = DiabeticRetinopathyModel(num_classes=3)


    # Training the dataset with ResNet50
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move model to device
    model.to(device)

    # Training loop
    num_epochs = 100
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_dataloader)
        val_epoch_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}%, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc*100:.2f}%")
        
        if best_acc < val_epoch_acc:
            torch.save(model, "./fine_tuned_resnet152.pth")
            best_acc = val_epoch_acc
        print(" ")

    print("Complete !!")    
    print(f"Your Best Model Acc is : {best_acc*100:.2f}%")