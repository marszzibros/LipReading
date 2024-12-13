import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np


class SentenceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.frame_data = pd.read_csv(csv_file, index_col=0)
        self.word_list = []
        with open("selected_words.txt", "r") as words:
            for word in words:
                self.word_list.append(word.strip())
        self.word_list.append("[MASK]")

        self.word_dict = {i: v for i, v in enumerate(self.word_list)}
        self.word_rev_dict = {v: i for i, v in enumerate(self.word_list)}

        self.transform = transform

    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_path = "../" + self.frame_data.iloc[idx]['vid_path']

        # Load video frames
        cap = cv2.VideoCapture(video_path)
        counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sentence_tensor = self.frame_data.iloc[idx]['stem_sen'].split(" ")
        for i, word in enumerate(sentence_tensor):
            if word not in self.word_list:
                sentence_tensor[i] = self.word_rev_dict["[MASK]"]
            else:
                sentence_tensor[i] = self.word_rev_dict[word]

        frames = []
        for frame_idx in range(counts):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx} from video {video_path}")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        # Apply sliding window technique
        window_size = 9
        step = 1

        sliding_windows = [
            frames[i:i + window_size]
            for i in range(0, len(frames) - window_size + 1, step)
        ]

        # Convert sliding windows to tensors
        sliding_windows_tensors = torch.stack([
            torch.stack(window) for window in sliding_windows
        ])

        return sliding_windows_tensors, sentence_tensor
