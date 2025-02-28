import os
import re
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class CoinRunDataset(Dataset):
    def __init__(self, root_dir, seq_len=8, frame_size=(128, 128)):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
        ])
        self.sequences = []
        pattern = re.compile(r"frame_(\d+)_(\d+)\.png")
        self.episodes = {}
        
        if not os.path.exists(root_dir):
            print(f"Warning: Directory {root_dir} does not exist. Creating empty dataset.")
            return
            
        for fname in os.listdir(root_dir):
            m = pattern.match(fname)
            if m:
                ep, fr = int(m.group(1)), int(m.group(2))
                self.episodes.setdefault(ep, []).append((fr, fname))
        
        for ep in self.episodes:
            frames = sorted(self.episodes[ep], key=lambda x: x[0])
            for start in range(len(frames) - self.seq_len):
                self.sequences.append((ep, start))
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ep, start = self.sequences[idx]
        input_frames = self.episodes[ep][start:start + self.seq_len]
        
        input_imgs = [self.transform(Image.open(os.path.join(self.root_dir, f[1]))) for f in input_frames]
        
        frames_tensor = torch.stack(input_imgs).to(torch.float16)  # Use half precision
        
        return frames_tensor