import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, re
from PIL import Image

from lam import LatentActionModel
from dynamics import DynamicsModel
from frame_dataset import CoinRunDataset

from vqvae import VQVAE

config = {
    'batch_size': 8,
    'seq_len': 16,             # Number of frames per sequence
    'latent_dim': 32,
    'num_actions': 8,           # 8 discrete action codes
    'num_layers': 6,            # Number of transformer layers
    'n_head': 8,                # Number of attention heads
    'dropout': 0.1,
    'lr': 3e-4,
    'num_epochs': 50,
    'frame_size': (128, 128),
    # Settings for LAM’s convolutional encoder/decoder
    'encoder_d_model': 32,
    'encoder_num_layers': 2,
    'decoder_d_model': 256,
    'decoder_num_layers': 2,
    'num_downsampling_layers': 2,
    'num_upsampling_layers': 7,
    'num_residual_layers': 2,
    'num_residual_hiddens': 64,
    'num_tokens': 1024,
}

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # just args for pretrained video tokenizer
    tok_args = {
        "in_channels": 3,
        "num_hiddens": 512,
        "num_downsampling_layers": 2,
        "num_residual_layers": 6,
        "num_residual_hiddens": 64,
        "embedding_dim": 4,
        "num_embeddings": 1024,
        "use_ema": True,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    

    lam = LatentActionModel(config).to(device)
    dm = DynamicsModel(config).to(device)
    tokenizer = VQVAE(**tok_args).to(device)

    #checkpoint = torch.load('model.pth', map_location=device)
    #tokenizer.load_state_dict(checkpoint)

    optimizer = optim.Adam(list(lam.parameters()) + list(dm.parameters()), lr=config['lr'])

    mse_loss = nn.MSELoss()       
    ce_loss = nn.CrossEntropyLoss()  

    dataset = CoinRunDataset(root_dir='coinrun_frames',
                                    seq_len=config['seq_len'],
                                    frame_size=config['frame_size'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    lam_loss_epoch = 0.0
    dm_loss_epoch = 0.0

    for epoch in range(config['num_epochs']):
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)  # [B, T, C, H, W]
            batch = batch.permute(0,2,1,3,4)
            B, C, T, H, W = batch.shape
            
            with torch.no_grad():
                tokens = tokenizer(batch)["embed"]  # [B, T, H_token, W_token]
                tokens = tokens.flatten(start_dim=2)  # [B, T, num_tokens]
            
            optimizer.zero_grad()
            lam_losses = []
            dm_losses = []
            
            for t in range(config['seq_len'], T):
                input_frames = batch[:, t-config['seq_len']:t]  # [B, seq_len, C, H, W]
                target_frame = batch[:, :, t].unsqueeze(2) # [B, 1, C, H, W]
                recon_frame, actions = lam(input_frames, target_frame)
                actions = torch.stack(actions, dim=1)  # [B, seq_len]
                
                input_tokens = tokens[:, t-config['seq_len']:t - 1]  # [B, seq_len, num_tokens]
                target_tokens = tokens[:, t - 1]  # [B, num_tokens]
                
                loss_lam = mse_loss(recon_frame, target_frame)
                lam_losses.append(loss_lam)

                logits = dm(input_tokens, actions)
                loss_dm = ce_loss(logits.view(-1, 1024), target_tokens.view(-1))
                dm_losses.append(loss_dm)
                
            total_loss = torch.mean(torch.stack(dm_losses)) + torch.mean(torch.stack(lam_losses))
            total_loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} | LAM Loss: {torch.stack(lam_losses).item():.4f} | DM Loss: {torch.stack(dm_losses).item():.4f}")
        print(f"Epoch {epoch} completed. Combined Loss: {total_loss.item():.4f}")

if __name__ == '__main__':
    train()