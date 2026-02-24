# Download the dataset
# This cell has to run only once. 
# NO need to run every time you arrive on this notebook. 

import requests
import tarfile
import os
import shutil

# Define the URL and folder paths
url = "https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
folder_name = "flowers"
file_name = "flower_data.tar.gz"
file_path = os.path.join(folder_name, file_name)

# Remove the folder or symbolic link if it already exists (equivalent to `rm -rf flowers`)
try:
    if os.path.islink(folder_name) or os.path.isfile(folder_name):
        os.remove(folder_name)  # Remove the symbolic link or file
    elif os.path.isdir(folder_name):
        shutil.rmtree(folder_name)  # Remove the directory
    print(f"Removed existing {folder_name} folder/file/soft link, if any.")
except FileNotFoundError:
    pass  # If the file or directory does not exist, do nothing

# Create the folder
os.makedirs(folder_name)
print(f"Created folder: {folder_name}")

# Download the file
response = requests.get(url, stream=True)

# Save the file in the 'flowers' folder
with open(file_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)

print(f"Downloaded {file_name} to {folder_name}")

# Extract the file in the 'flowers' folder
if file_path.endswith("tar.gz"):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=folder_name)
        print(f"Extracted {file_name} to {folder_name}")

# Clean up by removing the tar.gz file after extraction
os.remove(file_path)
print(f"Removed the downloaded tar.gz file: {file_path}")

# Imports here
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
# Define transformations for training (with augmentation)
data_transforms_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define transformations for validation and test (without augmentation)
data_transforms_val_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = data_transforms_train)
val_dataset = datasets.ImageFolder(valid_dir, transform = data_transforms_val_test)
test_dataset = datasets.ImageFolder(test_dir, transform = data_transforms_val_test)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 64)
test_loader = DataLoader(test_dataset, batch_size = 64)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Quick check: How many categories do we have?
print(f"Number of categories: {len(cat_to_name)}")

# TODO: Build and train your network
# --- CONFIGURABLE HYPERPARAMETERS ---
arch = 'resnet50'      # Supported: 'resnet50', 'vgg16'
learning_rate = 0.001
hidden_units = 512     
epochs = 3
use_gpu = True         

# Set device automatically
device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

# 1. Load Pre-trained Network & Build Classifier
def setup_model(arch, hidden_units):
    if arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        input_features = model.fc.in_features
        for param in model.parameters():
            param.requires_grad = False
            
        model.fc = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        # Optimizer targets model.fc
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
        
    elif arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        input_features = 25088 
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        # Optimizer targets model.classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, optimizer

model, optimizer = setup_model(arch, hidden_units)
model.to(device)
criterion = nn.NLLLoss()

# 2. Training Logic
steps = 0
print_every = 20

print(f"Training started with {arch} on {device}...")

for epoch in range(epochs):
    running_loss = 0
    model.train()
    
    for inputs, labels in train_loader:
        steps += 1
        
        # CRITICAL: Move tensors to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            val_loss = 0
            accuracy = 0
            model.eval() 
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # CRITICAL: Move tensors to device
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)
                    val_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Val loss: {val_loss/len(val_loader):.3f}.. "
                  f"Val accuracy: {accuracy/len(val_loader):.3f}")
            
            running_loss = 0
            model.train()
            
# TODO: Do validation on the test set
# Function to test the model and calculate accuracy
def check_test_accuracy(model, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')

check_test_accuracy(model, test_loader)

# TODO: Save the checkpoint 
model.class_to_idx = train_dataset.class_to_idx

checkpoint = {
    'arch': arch,
    'hidden_units': hidden_units,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    'optimizer_state': optimizer.state_dict(),
    'epochs': epochs
}

torch.save(checkpoint, 'resnet_checkpoint.pth')
print("Checkpoint saved! Path: resnet_checkpoint.pth")

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=device)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = checkpoint['fc']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

