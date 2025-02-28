import torch
import torch.nn as nn

class LatentActionModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, codebook_size, latent_dim):
        super(LatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        
        self.input_proj = nn.Linear(latent_dim, d_model)
        
        self.transformer_layers = nn.ModuleList([
            STAttentionBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        self.vq_vae = VQVAE(d_model, codebook_size)
        
        self.action_proj = nn.Linear(d_model, latent_dim)
        
    def forward(self, z_frames):
        batch_size, seq_len, _ = z_frames.shape
        
        x = self.input_proj(z_frames)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        action_encodings = x[:, :-1]  # [batch_size, seq_len-1, d_model]
        
        # Quantize action representations
        z_e, vq_loss, z_q = self.vq_vae(action_encodings)
        
        actions = self.action_proj(z_q)  # [batch_size, seq_len-1, latent_dim]
        
        return actions, z_e, z_q, vq_loss

class STAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(STAttentionBlock, self).__init__()
        self.spatial_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Spatial attention
        residual = x
        spatial_input = x.view(-1, x.size(-2), x.size(-1))
        x, _ = self.spatial_attn(spatial_input, spatial_input, spatial_input)

        if len(residual.shape) == 4:
            x = x.view(residual.size(0), residual.size(1), residual.size(2), residual.size(3))
        else:
            x = x.view(residual.size(0), residual.size(1), residual.size(2))
        x = self.norm1(x + residual)

        # Temporal attention (causal mask)
        residual = x
        seq_len = x.size(1)
        batch_size_spatial = x.size(0) * x.size(2)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)

        if len(x.shape) == 4:
            temporal_input = x.permute(0, 2, 1, 3).contiguous().view(-1, seq_len, x.size(-1))
            x, _ = self.temporal_attn(temporal_input, temporal_input, temporal_input, attn_mask=mask)
            x = x.view(residual.size(0), residual.size(2), residual.size(1), residual.size(3)).permute(0, 2, 1, 3).contiguous()
        else:
            temporal_input = x.view(-1, seq_len, x.size(-1))
            x, _ = self.temporal_attn(temporal_input, temporal_input, temporal_input, attn_mask=mask)
            x = x.view(residual.size(0), residual.size(1), residual.size(2))

        x = self.norm2(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.norm3(x + residual)
        x = self.dropout(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, latent_dim, codebook_size):
        super(VQVAE, self).__init__()
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim

    def forward(self, z_e):
        # z_e: (batch_size, sequence_length, latent_dim)
        original_shape = z_e.shape
        z_e_flat = z_e.reshape(-1, self.latent_dim)  

        distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_e_flat, self.embedding.weight.t())

        # Quantization
        encoding_indices = torch.argmin(distances, dim=1)
        z_q_flat = self.embedding(encoding_indices)
        z_q = z_q_flat.reshape(original_shape)  # Reshape

        commitment_loss = torch.mean((z_q.detach() - z_e)**2)
        codebook_loss = torch.mean((z_q - z_e.detach())**2)
        loss = commitment_loss + codebook_loss

        z_q = z_e + (z_q - z_e).detach()

        return z_e, loss, z_q