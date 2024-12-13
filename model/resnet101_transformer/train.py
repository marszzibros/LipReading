import torch
from torch import nn, optim
from model import LipReadingFactorized
from data_utils import LipReadingDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import logging

logging.basicConfig(filename='training_log.log', 
                    filemode='w',  # 'w' to overwrite log file each run, 'a' to append
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    level=logging.INFO)

logging.info("Starting Training")

csv_file = "../../data/metadata.csv"
num_classes = 301

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Instantiate the dataset
video_train = LipReadingDataset(csv_file=csv_file, transform=transform, data_type='train')
video_val   = LipReadingDataset(csv_file=csv_file, transform=transform, data_type='val')
video_test  = LipReadingDataset(csv_file=csv_file, transform=transform, data_type='test')

# Create dataloaders
trainloader = DataLoader(video_train, batch_size=8, shuffle=True, num_workers=8)
valloader = DataLoader(video_val, batch_size=8, shuffle=False, num_workers=8)
testloader = DataLoader(video_test, batch_size=8, shuffle=False, num_workers=8)

# Define the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ViTLipReadingFactorized()
model = model.to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-4)


os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)


num_epochs = 15
for epoch in range(num_epochs):
    logging.info(f'Starting Epoch {epoch+1}/{num_epochs}')
    
    # Training phase
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_fp = [0] * num_classes
    class_fn = [0] * num_classes
    
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Gradient Accumulation
        outputs = model(inputs)
        loss = criterion(outputs, labels) / 8
        loss.backward()

        # Step size 4
        if (i + 1) % 8 == 0 or i == len(trainloader) - 1: 
            optimizer.step()
            optimizer.zero_grad()

            # Log accuracy every 10 steps
            if (i + 1) % 50 == 0:
                logging.info(f"Step {i+1}, Accuracy: {total_correct / total_samples:.4f}")
        
        running_loss += loss.item() * inputs.size(0) * 8
        total_samples += labels.size(0)

        # Calculate metrics
        _, predicted = outputs.max(1)
        total_correct += (predicted == labels).sum().item()
        for c in range(num_classes):
            class_correct[c] += ((predicted == c) & (labels == c)).sum().item()
            class_total[c] += (labels == c).sum().item()
            class_fp[c] += ((predicted == c) & (labels != c)).sum().item()
            class_fn[c] += ((predicted != c) & (labels == c)).sum().item()
        

    # Calculate overall accuracy
    train_accuracy = total_correct / total_samples

    # Calculate per-class precision, recall, and F1 score
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    for c in range(num_classes):
        precision = class_correct[c] / (class_correct[c] + class_fp[c]) if (class_correct[c] + class_fp[c]) > 0 else 0.0
        recall = class_correct[c] / (class_correct[c] + class_fn[c]) if (class_correct[c] + class_fn[c]) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1_score)

    # Average precision, recall, and F1 score
    avg_train_precision = sum(precision_per_class) / num_classes
    avg_train_recall = sum(recall_per_class) / num_classes
    avg_train_f1 = sum(f1_per_class) / num_classes

    # Log training metrics
    epoch_loss = running_loss / len(trainloader.dataset)
    logging.info(f'Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}, '
                 f'Precision: {avg_train_precision:.4f}, Recall: {avg_train_recall:.4f}, F1 Score: {avg_train_f1:.4f}')

    # Validation phase
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_fp = [0] * num_classes
    class_fn = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            # Calculate metrics
            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            for c in range(num_classes):
                class_correct[c] += ((predicted == c) & (labels == c)).sum().item()
                class_total[c] += (labels == c).sum().item()
                class_fp[c] += ((predicted == c) & (labels != c)).sum().item()
                class_fn[c] += ((predicted != c) & (labels == c)).sum().item()

    # Calculate overall accuracy
    val_accuracy = total_correct / total_samples

    # Calculate per-class precision, recall, and F1 score
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    for c in range(num_classes):
        precision = class_correct[c] / (class_correct[c] + class_fp[c]) if (class_correct[c] + class_fp[c]) > 0 else 0.0
        recall = class_correct[c] / (class_correct[c] + class_fn[c]) if (class_correct[c] + class_fn[c]) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1_score)

    # Average precision, recall, and F1 score
    avg_val_precision = sum(precision_per_class) / num_classes
    avg_val_recall = sum(recall_per_class) / num_classes
    avg_val_f1 = sum(f1_per_class) / num_classes

    # Log validation metrics
    epoch_val_loss = val_loss / len(valloader.dataset)
    logging.info(f'Epoch {epoch+1} - Val Loss: {epoch_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, '
                 f'Precision: {avg_val_precision:.4f}, Recall: {avg_val_recall:.4f}, F1 Score: {avg_val_f1:.4f}')
    torch.save(model.state_dict(), f'logs/ep{epoch+1}resnet34_3.pth')
logging.info("Training Completed")

print("Model saved.")
