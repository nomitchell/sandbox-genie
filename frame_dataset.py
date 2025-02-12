import os
import re
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms

class SequentialFrameDataset(Dataset):
    def __init__(self, root_dir, clip_length=16, transform=None):
        """
        Args:
            root_dir (str): Path to the directory containing all frames.
            clip_length (int): Number of frames per clip.
            transform (callable, optional): Transform to apply to the frames.
        """
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform

        # Get sorted list of all image file paths
        self.frames = sorted(
            [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
        )

    def __len__(self):
        return len(self.frames) - self.clip_length + 1

    def __getitem__(self, idx):
        clip_paths = self.frames[idx : idx + self.clip_length]
        frames = [Image.open(p).convert("RGB") for p in clip_paths]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames into a tensor of shape (C, T, H, W)
        frames = torch.stack(frames, dim=0)  # Shape: (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # Shape: (C, T, H, W)

        return frames

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
        for fname in os.listdir(root_dir):
            m = pattern.match(fname)
            if m:
                ep, fr = int(m.group(1)), int(m.group(2))
                self.episodes.setdefault(ep, []).append((fr, fname))
        for ep in self.episodes:
            frames = sorted(self.episodes[ep], key=lambda x: x[0])
            for start in range(len(frames) - self.seq_len):
                # Input: seq_len frames, Target: start + seq_len frame
                self.sequences.append((ep, start))
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ep, start = self.sequences[idx]
        input_frames = self.episodes[ep][start:start + self.seq_len + 1]
        #target_frame = self.episodes[ep][start + self.seq_len]
        # Load and transform frames
        input_imgs = [self.transform(Image.open(os.path.join(self.root_dir, f[1]))) for f in input_frames]
        #target_img = self.transform(Image.open(os.path.join(self.root_dir, target_frame[1])))
        return torch.stack(input_imgs) #, target_img