import torch
from torch import nn, optim
from model import LipReadingFactorized
from data_utils import LipReadingDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd

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

video_test  = LipReadingDataset(csv_file=csv_file, transform=transform, data_type='test')

testloader = DataLoader(video_test, batch_size=8, shuffle=False, num_workers=8)

# Define the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = LipReadingFactorized()
checkpoint = torch.load('logs/ep15resnet34_3.pth')
model.load_state_dict(checkpoint)
model = model.to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_classes = 301
# Initialize metrics
class_correct = [0] * num_classes
class_total = [0] * num_classes
class_fp = [0] * num_classes
class_fn = [0] * num_classes
total_correct = 0
total_samples = 0
val_loss = 0

# Validation loop
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * inputs.size(0)
        
        # Predictions and Metrics
        _, predicted = outputs.max(1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        for c in range(num_classes):
            class_correct[c] += ((predicted == c) & (labels == c)).sum().item()
            class_total[c] += (labels == c).sum().item()
            class_fp[c] += ((predicted == c) & (labels != c)).sum().item()
            class_fn[c] += ((predicted != c) & (labels == c)).sum().item()


# Calculate hit rate
hit_rate = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
word_list = []
with open("selected_words.txt", "r") as words:
    for word in words:
        word_list.append(word.strip())
word_list.append("[MASK]")
# Create a DataFrame
data = {
    "Word": word_list,
    "Hits": class_correct,
    "Non-hits": class_fn,
    "Total": class_total,
    "Hit Rate": hit_rate,
}
df = pd.DataFrame(data)

# Save DataFrame to CSV
csv_file_path = "word_classification_metrics.csv"
df.to_csv(csv_file_path, index=False)
print(f"Metrics saved to {csv_file_path}")

# Plotting Hits vs Non-hits
plt.figure(figsize=(12, 6))
x = range(num_classes)
width = 0.4

# Bar chart for hits and non-hits
plt.bar(x, class_correct, width=width, label="Hits", alpha=0.8, color='blue')
plt.bar([p + width for p in x], class_fn, width=width, label="Non-hits", alpha=0.8, color='orange')

# Customizing the plot
plt.xticks([p + width / 2 for p in x], word_list, rotation=45, ha='right')
plt.xlabel("Word Classes")
plt.ylabel("Count")
plt.title("Hits and Non-hits for Word Classes")
plt.legend()
plt.tight_layout()

# Save the plot
plot_file_path = "word_classification_plot.png"
plt.savefig(plot_file_path)
