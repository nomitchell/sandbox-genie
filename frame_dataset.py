import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as T

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
        # Number of possible clips
        return len(self.frames) - self.clip_length + 1

    def __getitem__(self, idx):
        # Get paths for the clip
        clip_paths = self.frames[idx : idx + self.clip_length]
        # Load and transform frames
        frames = [Image.open(p).convert("RGB") for p in clip_paths]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames into a tensor of shape (C, T, H, W)
        frames = torch.stack(frames, dim=0)  # Shape: (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # Shape: (C, T, H, W)

        return frames
'''
# Parameters
root_dir = "path/to/frames"  # Folder with all images
clip_length = 16  # Number of frames per clip
batch_size = 8  # Number of clips per batch
transform = T.Compose([
    T.Resize((128, 128)),  # Resize frames
    T.ToTensor(),          # Convert to tensors
])

# Dataset and DataLoader
dataset = SequentialFrameDataset(root_dir, clip_length, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Example: Iterate through the DataLoader
for batch in dataloader:
    print(batch.shape)  # (B, C, T, H, W) where B=batch_size
    break'''
