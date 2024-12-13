import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from nltk.stem import PorterStemmer
import spacy

class LipReadingDataset(Dataset):
    def __init__(self, csv_file, transform=None, data_type="train", word_limit=300, save_path="selected_words.txt"):

        # Load frame data and filter by data type (train, test, etc.)
        self.frame_data = pd.read_csv(csv_file, index_col=0)
        self.frame_data = self.frame_data[self.frame_data['type'] == data_type]

        # Randomize the rows for generalization
        self.frame_data = self.frame_data.sample(frac=1, random_state=42)
        
        self.stemmer = PorterStemmer()
        nlp = spacy.load("en_core_web_sm")

        

        self.frame_data['word'] = self.frame_data['word'].apply(lambda x: self.stemmer.stem(x))
        word_list = list(set(self.frame_data['word']))
        
        doc = nlp(" ".join(word_list))
        # Filter out tokens that are named entities
        filtered_words = [token.text for token in doc if token.ent_type_ == ""]
        
        self.frame_data = self.frame_data[self.frame_data['word'].apply(lambda x : self.stemmer.stem(x) in filtered_words)]

        selected_words = random.sample(filtered_words, min(word_limit, len(word_list))) 
        if not os.path.exists(save_path):
            with open(save_path, "w") as f:
                for word in selected_words:
                    f.write(f"{word}\n")
        else:
            with open(save_path, "r") as f:
                selected_words = [line.strip() for line in f]

        self.frame_data_inwords = self.frame_data[self.frame_data['word'].apply(lambda x : self.stemmer.stem(x) in selected_words)]
        self.frame_data_notinwords = self.frame_data[~self.frame_data['word'].apply(lambda x : self.stemmer.stem(x) in selected_words)]

        sample_per_word = len(self.frame_data_inwords[self.frame_data_inwords['word'].apply(lambda x: self.stemmer.stem(x) == selected_words[0])])
        self.frame_data_notinwords = self.frame_data_notinwords.sample(
            n=sample_per_word, 
            random_state=42
        )
        self.frame_data_notinwords.loc[:, 'word'] = "[MASK]"
        self.frame_data = pd.concat([self.frame_data_inwords, self.frame_data_notinwords])
        
        self.word_dict = {i: v for i, v in enumerate(selected_words + ['[MASK]'])}
        self.word_rev_dict = {v: i for i, v in enumerate(selected_words + ['[MASK]'])}

        self.transform = transform

    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load video path and frame count
        video_path = self.frame_data.iloc[idx]['vid_path']
        counts = self.frame_data.iloc[idx]['frame_count']

        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        for frame_idx in range(counts - 2):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx} from video {video_path}")
                break
            if frame_idx % 3 == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
        cap.release()

        # Stack frames into a tensor
        frames_tensor = torch.stack(frames)

        # Retrieve the label for the word
        word_tensor = self.word_rev_dict[self.frame_data.iloc[idx]['word']]

        return frames_tensor, word_tensor
