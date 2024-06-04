import time
import os
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms, ToTensor
from torch.utils.data import DataLoader, random_split
from botanist import (LeafDataset,DataLoader, Dataset, model, 
                      device, device, test_loader, dataset, checkCudaAvaiable)

checkCudaAvaiable()

class ValLeafDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]  # Return the image and its filename

model_path = 'trained_models/cnn_35k_98.81.pth'

if __name__ == "__main__":
    # model = LeafCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    print("start to evaluate ...")
    start = time.time()
    # Create the dataset and data loader for the validation set
    val_dataset = ValLeafDataset(root_dir='TestFiles/', transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    predictions = []

    # Make predictions on the validation set
    with torch.no_grad():
        for images, filenames in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            predicted = predicted.cpu().numpy()  # np can only run on CPU

            filenames = [str(f).replace('.jpg', '') for f in filenames]    # Ensure filenames are in a plain text format
            for filename, label in zip(filenames, predicted):
                predictions.append((filename, label + 1))     # Adjust back to 1-38 for the output
    
    print(f"Testset size: {len(val_dataset)}, Elapsed time: {str(round(time.time()-start, 3))}")
    # Create DataFrame and save to CSV

    predictions_df = pd.DataFrame(predictions, columns=['filename', 'label'])
    predictions_df.to_csv('val_predictions2.csv', index=False)
