import os
import torch
import pygame
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.cuda.amp as amp

from blocks.vt import STTransformer, VQVAE
from blocks.lam import LatentActionModel
from blocks.dyn import DynamicsModel

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

class WorldModelTest:
    def __init__(self, checkpoint_path="checkpoints"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.video_tokenizer = STTransformer(**video_tokenizer_params).to(self.device)
        self.action_model = LatentActionModel(**action_model_params).to(self.device)
        self.dynamics_model = DynamicsModel(**dynamics_model_params).to(self.device)
        
        self.video_tokenizer.load_state_dict(torch.load(f"{checkpoint_path}/video_tokenizer.pth"))
        self.action_model.load_state_dict(torch.load(f"{checkpoint_path}/action_model.pth"))
        self.dynamics_model.load_state_dict(torch.load(f"{checkpoint_path}/dynamics_model.pth"))
        
        self.video_tokenizer.eval()
        self.action_model.eval()
        self.dynamics_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def autonomous_rollout(self, initial_frame, num_steps=50):
        """Generate autonomous trajectory from initial frame"""
        with torch.no_grad():
            if isinstance(initial_frame, Image.Image):
                initial_frame = self.transform(initial_frame).unsqueeze(0)
            
            _, z_e_initial, _, _ = self.video_tokenizer(initial_frame)
            trajectory = [z_e_initial]
            
            for _ in range(num_steps):
                z_trajectory = torch.stack(trajectory, dim=1)
                actions, _, _, _ = self.action_model(z_trajectory)
                
                mean, log_var = self.dynamics_model(trajectory[-1], actions[:, -1:])
                std = torch.exp(0.5 * log_var)
                next_latent = mean + std * torch.randn_like(std)
                trajectory.append(next_latent)
            
            full_trajectory = torch.stack(trajectory, dim=1)
            frames, _, _, _ = self.video_tokenizer(full_trajectory)
            return frames

    def interactive_play(self):
        pygame.init()
        screen = pygame.display.set_mode((128, 128))
        clock = pygame.time.Clock()
        
        current_frame = Image.open("coinrun_frames/frame_0.png")
        current_frame_tensor = self.transform(current_frame).unsqueeze(0)
        
        running = True
        trajectory = []
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            keys = pygame.key.get_pressed()
            action = torch.zeros(1, 1, action_model_params["latent_dim"]).to(self.device)
            
            # Temporary action encoding, need to change when actual models are trained
            if keys[pygame.K_LEFT]:
                action[0, 0, 0] = -1  
            elif keys[pygame.K_RIGHT]:
                action[0, 0, 0] = 1
            elif keys[pygame.K_SPACE]:
                action[0, 0, 1] = 1
            
            with torch.no_grad():
                _, z_e, _, _ = self.video_tokenizer(current_frame_tensor)
                
                if z_e.dim() != action.dim():
                    z_e = z_e.unsqueeze(1)
                mean, log_var = self.dynamics_model(z_e, action)
                std = torch.exp(0.5 * log_var)
                next_latent = mean + std * torch.randn_like(std)
                
                next_frame, _, _, _ = self.video_tokenizer(next_latent)
                
                frame_np = next_frame.squeeze().permute(1, 2, 0).cpu().numpy()
                frame_surface = pygame.surfarray.make_surface(frame_np * 255)
                screen.blit(frame_surface, (0, 0))
                pygame.display.flip()
                
                current_frame_tensor = next_frame
            
            clock.tick(30)
        
        pygame.quit()

if __name__ == "__main__":
    world_model = WorldModelTest()
    
    initial_frame = Image.open("coinrun_frames/frame_0.png")
    generated_frames = world_model.autonomous_rollout(initial_frame)
    
    for i, frame in enumerate(generated_frames.squeeze()):
        frame_img = transforms.ToPILImage()(frame)
        frame_img.save(f"generated_frames/frame_{i}.png")
    
    world_model.interactive_play()