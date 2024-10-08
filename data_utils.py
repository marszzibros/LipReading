import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class LipReadingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations or frame metadata.
            root_dir (str): Directory with all the video frame folders.
            transform (callable, optional): Optional transform to be applied on a frame.
        """
        self.frame_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.frame_data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Load additional information from CSV if available
        label = self.frame_data.iloc[idx, 1]  # Assuming the second column is a label
        
        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 'label': label}

        return sample

# Example transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Instantiate the dataset
csv_file = '/users/j/j/jjung2/scratch/data/subset_test.csv'
root_dir = 'path/to/your/dataset'
video_dataset = LipReadingDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

# Create a dataloader
dataloader = DataLoader(video_dataset, batch_size=32, shuffle=True, num_workers=4)
