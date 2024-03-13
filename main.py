import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from torchvision.models import resnet50, ResNet50_Weights
from dataset_processor import GenericSimpleOneHotDataset

torch.set_num_threads(2)

if __name__ == '__main__' :

    total_epochs = 10
    batch_size   = 10

    model_weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights = model_weights)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 3),
        torch.nn.Sigmoid()
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
    lossFunction = torch.nn.CrossEntropyLoss()

    transform_functions = model_weights.transforms()

    validation_dataset = GenericSimpleOneHotDataset('./Dataset/validation', augmentation = transform_functions)
    training_dataset = GenericSimpleOneHotDataset('./Dataset/train', augmentation = transform_functions)
    testing_dataset = GenericSimpleOneHotDataset('./Dataset/test', augmentation = transform_functions)

    print(len(validation_dataset), len(training_dataset), len(testing_dataset))
    
    #code from stackoverflow
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'

    validation_datasetloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 1, shuffle = True, num_workers = 1)
    training_datasetloader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True, num_workers = 1)
    testing_datasetloader = torch.utils.data.DataLoader(testing_dataset, batch_size = 1, shuffle = True, num_workers = 1)

    best_acc = 0.0 

    for epoch in range(total_epochs):
        print("Epoch :", epoch)
        
        # Training Loop
        model.train()  # set the model to train
        epoch_training_loss = []
        for (image, label) in tqdm(training_datasetloader):
            output = model(image)
            loss = lossFunction(output, label)
            loss.backward()
            optimizer.step()

            # logging goes here !
            epoch_training_loss.append(loss.item())
        print("Average Loss :", np.mean(epoch_training_loss))

        # Validation Loop
        model.eval()
        correct_counter = 0
        false_counter   = 0
        for (image, label) in tqdm(validation_datasetloader):
            output = model(image)
            argmx = torch.argmax(output, dim = 1)
            oneHt = torch.zeros_like(output).scatter_(1, argmx.unsqueeze(1), 1.)
            if torch.equal(oneHt, label):
                correct_counter += 1
            else:
                false_counter += 1
        
        acuurracy = correct_counter / (correct_counter + false_counter)
        print(f"Correct : {correct_counter}, False : {false_counter}, Acc: {(acuurracy * 100):.2f}")
        
        if best_acc < acuurracy:
            torch.save(model.state_dict(), "./bestmodel.pth")
            best_acc = acuurracy
        print(" ")

    print("Complete !!")    
    print(f"Your Best Model Acc is : {best_acc:.2f}%")

    model.load_state_dict(torch.load('./bestmodel.pth'))
    model.eval()
    correct_counter = 0
    false_counter   = 0
    for (image, label) in tqdm(testing_datasetloader):
        output = model(image)
        argmx = torch.argmax(output, dim = 1)
        oneHt = torch.zeros_like(output).scatter_(1, argmx.unsqueeze(1), 1.)
        if torch.equal(oneHt, label):
            correct_counter += 1
        else:
            false_counter += 1

    acuurracy = correct_counter / (correct_counter + false_counter)
    print(f"Correct : {correct_counter}, False : {false_counter}, Acc: {(acuurracy * 100):.2f}")
    