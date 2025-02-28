import os
import multiprocessing

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.checkpoint import checkpoint

from utils.dataset import CoinRunDataset
from blocks.dyn import DynamicsModel
from blocks.vt import STTransformer
from blocks.lam import LatentActionModel

def train_video_tokenizer(model, dataloader, optimizer, epochs=10, accumulation_steps=6):
    # Create validation dataloader for visualization
    val_dataset = CoinRunDataset(root_dir="val_coinrun_frames", seq_len=seq_len, frame_size=(height, width))
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Get a fixed batch for visualization
    val_iter = iter(val_dataloader)
    try:
        vis_batch = next(val_iter)
    except StopIteration:
        print("Warning: Validation dataset is empty. Visualization will be skipped.")
        vis_batch = None
    
    # Create output directory for visualizations
    os.makedirs("visualizations", exist_ok=True)
    
    scaler = amp.GradScaler()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            video_frames = batch.to(device)
            with amp.autocast():
                reconstructed, z_e, z_q, vq_loss = model(video_frames)
                recon_loss = torch.mean((reconstructed - video_frames) ** 2)
                loss = recon_loss + vq_loss
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
            
            total_loss += loss.item() * accumulation_steps  # Scale back for reporting
            
            if ((i + 1) % 5) == 0:
                avg_loss = total_loss / (i + 1)
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Video Tokenizer Loss: {avg_loss:.6f}")
                
                # Generate visualization using the validation batch
                if vis_batch is not None:
                    visualize_reconstructions(epoch, i, vis_batch, model, device)
        
        epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete - Video Tokenizer Loss: {epoch_loss:.6f}")

        # Add to training loops
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/video_tokenizer_epoch{epoch+1}.pth")

def train_action_dynamics(action_model, dynamics_model, tokenizer, dataloader, optimizer_action, optimizer_dynamics, epochs=10, accumulation_steps=6):
    scaler = amp.GradScaler()
    action_model.train()
    dynamics_model.train()
    tokenizer.eval()  # Set to eval mode
    
    # Pre-compute latents for the entire dataset
    print("Pre-computing frame latents...")
    frame_latents = []
    
    with torch.no_grad():
        for batch in dataloader:
            video_frames = batch.to(device, non_blocking=True)
            with amp.autocast():
                _, z_e, _, _ = tokenizer(video_frames)
            frame_latents.append(z_e.cpu())
    
    # Free up tokenizer memory
    del tokenizer
    torch.cuda.empty_cache()
    
    # Now train with pre-computed latents
    for epoch in range(epochs):
        total_action_loss = 0
        total_dynamics_loss = 0
        optimizer_action.zero_grad()
        optimizer_dynamics.zero_grad()
        
        for i, latent_batch in enumerate(frame_latents):
            # Move latents back to GPU for this batch
            z_e = latent_batch.to(device)
            
            with amp.autocast():
                # Infer actions from full sequence
                actions, z_e_action, z_q_action, action_vq_loss = action_model(z_e)
                
                # Predict next latents using current latents and inferred actions
                mean, log_var = dynamics_model(z_e[:, :-1], actions)
                
                # Compute dynamics loss
                dynamics_loss = F.gaussian_nll_loss(
                    mean, 
                    z_e[:, 1:],  # Target is next frame
                    torch.exp(log_var),
                    full=True
                )
                
                # Total loss
                loss = dynamics_loss + action_vq_loss
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(frame_latents):
                scaler.step(optimizer_action)
                scaler.step(optimizer_dynamics)
                scaler.update()
                optimizer_action.zero_grad()
                optimizer_dynamics.zero_grad()
            
            total_action_loss += action_vq_loss.item() * accumulation_steps
            total_dynamics_loss += dynamics_loss.item() * accumulation_steps
            
            # Free up memory
            del z_e
            torch.cuda.empty_cache()
        
        # Print epoch statistics
        avg_action_loss = total_action_loss / len(frame_latents)
        avg_dynamics_loss = total_dynamics_loss / len(frame_latents)
        print(f"Epoch {epoch+1} - Action Loss: {avg_action_loss:.6f}, Dynamics Loss: {avg_dynamics_loss:.6f}")

def visualize_reconstructions(epoch, batch_idx, val_batch, model, device):
    """Generate and save visualizations of model reconstructions"""
    model.eval()
    
    with torch.no_grad():
        # Get input frames from validation batch
        input_frames = val_batch[0].to(device)
        batch_size, seq_len, channels, height, width = input_frames.shape
        
        # Generate reconstructions
        reconstructed, _, _, _ = model(input_frames)
        
        # Create visualization grid
        # We'll show original frames and their reconstructions
        grid_images = []
        
        # Add a separator column
        separator = torch.ones((batch_size, 1, channels, height, width), device=device) * 0.5
        
        # Format for visualization: original frames + separator + reconstructed frames
        grid_data = torch.cat([input_frames, separator, reconstructed], dim=1)
        
        # Create grid for each item in batch
        for b in range(batch_size):
            # Rearrange to (num_frames*2+1, channels, height, width)
            sample_frames = grid_data[b]
            grid = make_grid(sample_frames, nrow=(seq_len*2+1), normalize=True, pad_value=1)
            grid_images.append(grid)
        
        # Combine all batches into one large grid
        final_grid = make_grid(grid_images, nrow=1, normalize=False, pad_value=1)
        
        # Convert to image and save
        grid_np = final_grid.permute(1, 2, 0).numpy()
        grid_np = (grid_np * 255).astype(np.uint8)
        img = Image.fromarray(grid_np)
        
        # Save the visualization
        save_path = f"visualizations/epoch{epoch+1}_batch{batch_idx+1}_reconstructions.png"
        img.save(save_path)
        print(f"Saved visualization to {save_path}")
    
    # Set model back to training mode
    model.train()

# Wrap all the execution code in an if __name__ == '__main__': block
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # Use reduced parameters
    video_tokenizer_params = {
        "num_layers": 4,
        "d_model": 256,
        "num_heads": 4,
        "codebook_size": 512,
        "patch_size": 4,
        "latent_dim": 32
    }
    action_model_params = {
        "num_layers": 4,
        "d_model": 256,
        "num_heads": 4,
        "codebook_size": 6,
        "latent_dim": 32
    }
    dynamics_model_params = {
        "num_layers": 6,
        "d_model": 256,
        "num_heads": 4,
        "latent_dim": 32,
        "maskgit_steps": 10,
        "sampling_temperature": 1.0
    }
    optimizer_params = {
        "lr": 1e-4, 
        "weight_decay": 1e-5
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seq_len = 16
    channels = 3
    height = 128
    width = 128
    num_workers = 2

    dataset = CoinRunDataset(root_dir="coinrun_frames", seq_len=seq_len, frame_size=(height, width))

    # 1. Train video tokenizer with smaller batch size
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    steps_per_epoch = len(dataloader)
    video_tokenizer_epochs = 300000 // (steps_per_epoch * batch_size // 8)  # Adjust epochs to maintain total steps
    
    video_tokenizer = STTransformer(**video_tokenizer_params).to(device)
    optimizer_video = optim.Adam(video_tokenizer.parameters(), **optimizer_params)
    train_video_tokenizer(video_tokenizer, dataloader, optimizer_video, epochs=video_tokenizer_epochs, accumulation_steps=6)
    
    # Save checkpoint
    torch.save(video_tokenizer.state_dict(), "checkpoints/video_tokenizer.pth")

    # Convert model to half precision
    video_tokenizer = video_tokenizer.half()

    # 2. Train dynamics and LAM with memory-efficient approach
    batch_size = 6
    dynamics_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    steps_per_epoch = len(dynamics_dataloader)
    dynamics_lam_epochs = 200000 // (steps_per_epoch * batch_size // 6)  # Adjust epochs
    
    action_model = LatentActionModel(**action_model_params).to(device)
    dynamics_model = DynamicsModel(**dynamics_model_params).to(device)
    optimizer_action = optim.Adam(action_model.parameters(), **optimizer_params)
    optimizer_dynamics = optim.Adam(dynamics_model.parameters(), **optimizer_params)
    
    train_action_dynamics(action_model, dynamics_model, video_tokenizer, 
                         dynamics_dataloader, optimizer_action, optimizer_dynamics, 
                         epochs=dynamics_lam_epochs, accumulation_steps=6)
    
    # Save checkpoints
    torch.save(action_model.state_dict(), "checkpoints/action_model.pth")
    torch.save(dynamics_model.state_dict(), "checkpoints/dynamics_model.pth")

    print("Training complete!")