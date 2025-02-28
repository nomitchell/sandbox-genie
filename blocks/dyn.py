import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, latent_dim, maskgit_steps, sampling_temperature):
        super(DynamicsModel, self).__init__()
        self.latent_dim = latent_dim
        
        self.input_proj = nn.Linear(latent_dim, d_model)
        
        self.transformer_layers = nn.ModuleList([
            STAttentionBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        self.prediction_head = nn.Linear(d_model, latent_dim * 2)
        
        self.maskgit_steps = maskgit_steps
        self.sampling_temperature = sampling_temperature

    def forward(self, current_latents, action_latents):
        combined = current_latents + action_latents
        
        x = self.input_proj(combined)
        
        for layer in self.transformer_layers:
            x = layer(x)
            
        prediction = self.prediction_head(x)
        mean, log_variance = torch.split(prediction, self.latent_dim, dim=-1)
        
        return mean, log_variance

    def sample(self, current_latents, action_latents):
        batch_size, seq_len, _ = current_latents.shape
        masked_latents = torch.zeros_like(current_latents).to(current_latents.device)
        mask = torch.ones_like(current_latents).to(current_latents.device)

        for _ in range(self.maskgit_steps):
            mean, log_variance = self.forward(masked_latents, action_latents)
            variance = torch.exp(0.5 * log_variance)
            distribution = torch.distributions.Normal(mean, variance)
            confidence = distribution.probs(current_latents) * mask

            # Unmask based on confidence
            num_unmask = int(self.latent_dim * seq_len * 0.2) #Example percentage to unmask.
            _, indices = torch.topk(confidence.view(batch_size, -1), num_unmask, dim=-1)
            unmask_indices = torch.stack([indices // self.latent_dim, indices % self.latent_dim], dim=-1)

            for b in range(batch_size):
                for s, l in unmask_indices[b]:
                    mask[b, s, l] = 0
                    masked_latents[b, s, l] = current_latents[b, s, l]

        return masked_latents

    def rollout(self, current_frame, action_idx, video_tokenizer, action_model):
        """Generate next frame given current frame and discrete action"""
        with torch.no_grad():
            _, z_e, _, _ = video_tokenizer(current_frame)
            
            action_emb = action_model.vq_vae.embedding(
                torch.tensor([action_idx]).to(current_frame.device)
            ).unsqueeze(0)
            
            mean, log_var = self(z_e, action_emb)
            
            std = torch.exp(0.5 * log_var) * self.sampling_temperature
            next_latent = mean + std * torch.randn_like(std)
            
            next_frame, _, _, _ = video_tokenizer(next_latent)
            return next_frame

class STAttentionBlock(nn.Module):
    # Same STAttentionBlock implementation, should probably share later
    def __init__(self, d_model, num_heads):
        super(STAttentionBlock, self).__init__()
        self.spatial_attn = nn.MultiheadAttention(d_model, num_heads)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x, _ = self.spatial_attn(x, x, x)
        x = self.norm1(x + residual)

        residual = x
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len))  # Causal mask
        x, _ = self.temporal_attn(x, x, x, attn_mask=mask)
        x = self.norm2(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.norm3(x + residual)
        return x