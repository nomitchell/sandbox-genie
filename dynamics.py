import torch
import torch.nn as nn

class DynamicsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['latent_dim']
        self.num_tokens = config['num_tokens']
        self.token_emb = nn.Embedding(self.num_tokens, self.latent_dim)
        self.action_emb = nn.Embedding(config['num_actions'], self.latent_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, self.latent_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=config['n_head'],
                dropout=config['dropout'],
                batch_first=True
            ),
            num_layers=config['num_layers']
        )
        self.head = nn.Linear(self.latent_dim, self.num_tokens)

    def forward(self, token_seq, action_indices, mask=None):
        B, seq_len, num_tokens = token_seq.shape

        token_emb = self.token_emb(token_seq)  # [B, seq_len, num_tokens, latent_dim]
        action_emb = self.action_emb(action_indices).unsqueeze(2)  # [B, seq_len, 1, latent_dim]
        action_emb = action_emb.expand(-1, -1, num_tokens, -1)

        combined = token_emb + action_emb  # [B, seq_len, num_tokens, latent_dim]

        if mask is not None:
            combined = torch.where(mask.unsqueeze(-1), self.mask_token, combined)

        combined = combined.view(B, seq_len * num_tokens, self.latent_dim)
        out = self.transformer(combined)  # [B, seq_len*num_tokens, latent_dim]
        next_frame_logits = self.head(out[:, -num_tokens:])  # [B, num_tokens, num_tokens]
        return next_frame_logits

    def iterative_inference(self, conditioning_tokens, action_indices, num_iterations=8):
        B = conditioning_tokens.size(0)

        next_tokens = torch.full((B, self.num_tokens), fill_value=0, dtype=torch.long, device=conditioning_tokens.device)

        token_mask = torch.ones(B, 1, self.num_tokens, dtype=torch.bool, device=conditioning_tokens.device)

        for i in range(num_iterations):
            logits = self.forward(next_tokens.unsqueeze(1), action_indices)  # [B, num_tokens, num_tokens]
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_tokens = torch.max(probs, dim=-1)  # [B, num_tokens]
            threshold = 0.9
            keep_mask = confidence > threshold
            next_tokens = torch.where(keep_mask, pred_tokens, next_tokens)
        return next_tokens