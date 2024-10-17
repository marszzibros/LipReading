import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class LipReadingDataset(Dataset):
    def __init__(self, csv_file, transform = None, data_type="train"):
        """
        Args:
            csv_file (str): Path to the csv file with annotations or frame metadata.
            root_dir (str): Directory with all the video frame folders.
            transform (callable, optional): Optional transform to be applied on a frame.
        """
        self.frame_data = pd.read_csv(csv_file, index_col=0)
        self.frame_data = self.frame_data[self.frame_data['type'] == data_type]
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        vid_folder = self.frame_data.iloc[idx]['out_dir']
        frame_files = sorted(os.listdir(self.frame_data.iloc[idx]['out_dir'])) 
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(vid_folder, frame_file)
            image = Image.open(frame_path).convert('RGB')  # Ensure all images are RGB
            
            if self.transform:
                image = self.transform(image)
            
            frames.append(image)
        
        frames_tensor = torch.stack(frames)
        word_tensor = torch.tensor(self.frame_data.iloc[idx]['word'])


        return frames_tensor, word_tensor

transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
# Instantiate the dataset


csv_file = '/users/j/j/jjung2/scratch/Deep_Learning_Course/data/subset_test.csv'
video_dataset = LipReadingDataset(csv_file=csv_file,transform=transform)

# Create a dataloader
dataloader = DataLoader(video_dataset, batch_size=128, shuffle=True, num_workers=16)
