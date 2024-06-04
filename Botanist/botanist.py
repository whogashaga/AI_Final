import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import time
from datetime import datetime

def checkCudaAvaiable():
    print(f"~ ~ ~ ~ ~ ~ ~ {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} ~ ~ ~ ~ ~ ~ ~")
    if torch.cuda.is_available():
        print("CUDA and NV GPU detected.")
    else:
        print("No CUDA or NV GPU detected. Check with $nvcc --version or $nvidia-smi")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeafDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data_frame.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = int(self.data_frame.iloc[idx, 1]) - 1 # Although labels are 1-38, should adjust (-1) to 0-37 for torch

        if self.transform:
            image = self.transform(image)

        return image, label, self.data_frame.iloc[idx, 0]  # Return filename

# Transformations
transform = transforms.Compose([
    # transforms.Resize((256, 256)),          # Resize images to 256x256
    # transforms.RandomHorizontalFlip(),      # Data Augmentation 1
    # transforms.RandomVerticalFlip(),        # Data Augmentation 2
    # transforms.RandomRotation((-45, 45)),   # Data Augmentation 3
    transforms.ToTensor()
])


num_epochs = 30
batch_size = 64
model_path = "trained_models/ep30_50k_lay4.pth"
predictions_path = 'predictionCsv/pred_ep30_50k_lay4.csv'

# Dataset
dataset = LeafDataset(csv_file='Botanist_Training_Set.csv', root_dir='TrainFiles/', transform=transforms.ToTensor())

train_size = 50000
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# CNN model
class LeafCNN(nn.Module):
    def __init__(self):
        super(LeafCNN, self).__init__()
        self.model = nn.Sequential(
            # 3 channels for RGB. output 32 filters. kernel size 3x3. the filter moves 2 pixel at a time. padding for input: (256+1+1) x (256+1+1)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   # Input: (3, 256, 256) , Output: (32, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                   # Input: (32, 256, 256) , Output: (32, 128, 128)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),   # Input: (32, 128, 128) , Output: (64, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                   # Input: (64, 128, 128) , Output: (64, 64, 64)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Input: (64, 64, 64) , Output: (128, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                   # Input: (128, 64, 64) , Output: (128, 32, 32)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # Input: (128, 32, 32) , Output: (256, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                   # Input: (256, 32, 32) , Output: (256, 16, 16)
            
            nn.Flatten(),                  # Input: (256, 32, 32) , Output: (256x16x16, ) = (65536, )  {flatten a 3D tensor into 1D tensor}
            nn.Dropout(0.5),               # dropout probability of 0.5 to reduce overfitting
            nn.Linear(256 * 16 * 16, 512), # Input: (65536, ) , Output: (512, )
            # nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 38)             # Input: (512, ) , Output: (38, ) 
        )

    def forward(self, x):
        return self.model(x)

model = LeafCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training Flow
if __name__ == "__main__" and torch.cuda.is_available():
    checkCudaAvaiable()
    total_start = time.time()
    start = time.time()

    print(f"start training... epoch: {num_epochs}, batch: {batch_size}, train: {train_size}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        end = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, time: {str(round(end-start, 2))}")
        start = end
    print(f"Total elapsed time: {str(round((end-total_start)/60, 2))} mins")
    
    torch.save(model.state_dict(), model_path) # Save the trained model

    # = = = = = EVALUATING MODEL = = = = = =
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    print("start to evaluate ...")


    start = time.time()
    predictions = []

    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # calculate Accuracy

            predicted = predicted.cpu().numpy()            # np can only run on CPU
            
            # Convert filenames from tensor to plain text
            filenames = [str(f.item()) for f in filenames]
            for filename, label in zip(filenames, predicted):
                predictions.append((filename, label + 1))  # Adjust back to 1-38 for the output
        
    print(f'correct: {correct}, total: {total}')
    print(f'Accuracy: {100 * correct / total}%')
    print(f"Elapsed time: {str(round(time.time()-start, 5))}")

    # Create DataFrame and save to CSV
    predictions_df = pd.DataFrame(predictions, columns=['filename', 'label'])
    predictions_df.to_csv(predictions_path, index=False)