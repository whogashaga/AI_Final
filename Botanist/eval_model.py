import time
import pandas as pd
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from botanist import (DataLoader, batch_size, model, 
                      model_path, device, device, 
                      test_loader, dataset, checkCudaAvaiable)

checkCudaAvaiable()

if __name__ == "__main__":
    # model = LeafCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    print("start to evaluate ...")

    test_size = 10000
    train_size = len(dataset) - test_size
    test_dataset, _ = random_split(dataset, [test_size, train_size])
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    predictions_df.to_csv('predictions.csv', index=False)