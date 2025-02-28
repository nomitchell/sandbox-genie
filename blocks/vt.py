import torch
import torch.nn as nn
import torch.nn.functional as F

class STTransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, codebook_size, patch_size, latent_dim, dropout=0.1):
        super(STTransformer, self).__init__()
        self.patch_size = patch_size
        self.embedding = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.transformer_layers = nn.ModuleList([
            STAttentionBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.vq_vae = VQVAE(latent_dim, codebook_size)
        self.decoder = nn.ConvTranspose2d(latent_dim, 3, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        if len(x.shape) == 4:  # [batch_size, seq_len, height, width] or [batch_size, channels, height, width]
            # Check if second dimension is likely channels (usually 3 for RGB)
            if x.shape[1] == 3:
                # This is [batch, channels, height, width]
                batch_size = x.shape[0]
                c = x.shape[1]
                h, w = x.shape[2], x.shape[3]
                x = x.unsqueeze(1)  # [batch_size, 1, channels, height, width]
                seq_len = 1
            else:
                # This is [batch_size, seq_len, height, width]
                batch_size, seq_len, h, w = x.shape
                x = x.unsqueeze(2)  # [batch_size, seq_len, 1, height, width]
                c = 1
        elif len(x.shape) == 5:  # [batch_size, seq_len, channels, height, width]
            batch_size, seq_len, c, h, w = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected 4D or 5D tensor.")
        
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.embedding(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, seq_len, -1, x.size(-1))
        for layer in self.transformer_layers:
            x = layer(x)

        z_e, vq_loss, z_q = self.vq_vae(x)
        z_q = z_q.view(batch_size * seq_len, -1, self.vq_vae.latent_dim) # (batch*seq, h'*w', latent_dim)
        z_q = z_q.permute(0, 2, 1).contiguous() # (batch*seq, latent_dim, h'*w')
        h_out = int(h / self.patch_size)
        w_out = int(w / self.patch_size)
        z_q = z_q.view(batch_size*seq_len, self.vq_vae.latent_dim, h_out, w_out)
        reconstructed = self.decoder(z_q)
        reconstructed = reconstructed.view(batch_size, seq_len, c, h, w)
        return reconstructed, z_e, z_q, vq_loss

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
        residual = x
        spatial_input = x.view(-1, x.size(-2), x.size(-1))
        x, _ = self.spatial_attn(spatial_input, spatial_input, spatial_input)
        x = x.view(residual.size(0), residual.size(1), residual.size(2), residual.size(3))
        x = self.norm1(x + residual)

        residual = x
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        x, _ = self.temporal_attn(x.permute(0,2,1,3).contiguous().view(-1, seq_len, x.size(-1)), x.permute(0,2,1,3).contiguous().view(-1, seq_len, x.size(-1)), x.permute(0,2,1,3).contiguous().view(-1, seq_len, x.size(-1)), attn_mask=mask)
        x = x.view(residual.size(0), residual.size(2), residual.size(1), residual.size(3)).permute(0,2,1,3).contiguous()
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
        # z_e: (batch_size, sequence_length, h'*w', latent_dim)
        batch_size, seq_len, spatial, _ = z_e.shape
        z_e_flat = z_e.view(-1, self.latent_dim)

        distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_e_flat, self.embedding.weight.t())

        encoding_indices = torch.argmin(distances, dim=1)
        z_q_flat = self.embedding(encoding_indices)
        z_q = z_q_flat.view(batch_size, seq_len, spatial, self.latent_dim)

        commitment_loss = torch.mean((z_q.detach() - z_e)**2)
        codebook_loss = torch.mean((z_q - z_e.detach())**2)
        loss = commitment_loss + codebook_loss

        z_q = z_e + (z_q - z_e).detach()

        return z_e, loss, z_q