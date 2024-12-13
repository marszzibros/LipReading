import torch
from torch import nn, optim
from model import LipReadingFactorized
from data_utils import SentenceDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import logging

logging.basicConfig(filename='training_log.log', 
                    filemode='w',  # 'w' to overwrite log file each run, 'a' to append
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    level=logging.INFO)

logging.info("Starting Training")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Instantiate the dataset
video_train = SentenceDataset(csv_file="../../data/sentence.csv", transform=transform)

trainloader = DataLoader(video_train, batch_size=1, shuffle=True, num_workers=4)
testloader = DataLoader(video_test, batch_size=1, shuffle=False, num_workers=4)

# Define the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = LipReadingFactorized()
checkpoint = torch.load('ep15resnet34_3.pth')
model.load_state_dict(checkpoint)

model = model.to(device)

model.eval()

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

with torch.no_grad():
    for inputs, labels in trainloader:
        inputs = torch.tensor(inputs).to(device)[0]

        for input,label in zip(inputs, labels):
            f, c, w, h = input.shape
            outputs = model(input.reshape(1, f, c, w, h))
            top_k = 5

            
            top_values, top_indices = torch.topk(outputs, top_k)

            print(video_train.word_dict[top_indices[0][0].item()])
            print(video_train.word_dict[top_indices[0][1].item()])

            print()
        for label in labels:
            print(video_train.word_dict[label.item()])    
        break


                
