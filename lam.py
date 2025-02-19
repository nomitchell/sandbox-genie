import torch
import torch.nn as nn

class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        layers = []
        for _ in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv3d(num_hiddens, num_residual_hiddens, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(num_residual_hiddens, num_hiddens, kernel_size=1)
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return torch.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens):
        super().__init__()
        conv = []
        current_channels = in_channels
        for i in range(num_downsampling_layers):
            out_channels = num_hiddens if i > 0 else num_hiddens // 2
            conv.append(nn.Conv3d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
            conv.append(nn.ReLU())
            current_channels = out_channels
        conv.append(nn.Conv3d(current_channels, num_hiddens, kernel_size=3, padding=1))
        self.conv = nn.Sequential(*conv)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        h = self.conv(x)
        return self.residual_stack(h)

class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_upsampling_layers, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.conv = nn.Conv3d(embedding_dim, num_hiddens, kernel_size=3, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)
        upconv = []
        for i in range(num_upsampling_layers):
            # Downstream layers reduce channels towards 3
            if i < num_upsampling_layers - 2:
                in_channels, out_channels = num_hiddens, num_hiddens
            elif i == num_upsampling_layers - 2:
                in_channels, out_channels = num_hiddens, num_hiddens // 2
            else:
                in_channels, out_channels = num_hiddens // 2, 3
            upconv.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            if i < num_upsampling_layers - 1:
                upconv.append(nn.ReLU())
        self.upconv = nn.Sequential(*upconv)

    def forward(self, x):
        h = self.conv(x)
        h = self.residual_stack(h)
        return self.upconv(h)
    
class LatentActionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = 3
        num_hiddens = config['encoder_d_model']
        self.encoder = Encoder(in_channels, num_hiddens,
                                    config['num_downsampling_layers'],
                                    config['encoder_num_layers'],
                                    config['num_residual_hiddens'])
        self.action_projector = nn.Linear(num_hiddens, config['latent_dim'])
        # Codebook: 8 discrete codes.
        self.codebook = nn.Embedding(config['num_actions'], config['latent_dim'])
        self.decoder = Decoder(config['latent_dim'], config['decoder_d_model'],
                                    config['num_upsampling_layers'],
                                    config['decoder_num_layers'],
                                    config['num_residual_hiddens'])
    def forward(self, x_t, x_t1):
        x_in = torch.cat([x_t, x_t1], dim=2)
        z_e = self.encoder(x_in)
        z_e_avg = torch.mean(z_e, dim=(2,3))
        z_e_proj = self.action_projector(z_e_avg)
        distances = torch.cdist(z_e_proj, self.codebook.weight)
        indices = torch.argmin(distances, dim=-1)    # shape: [B]
        z_q = self.codebook(indices)
        z_q = z_q.detach() + (z_e_proj - z_e_proj.detach())
        #z_q_reshaped = z_q.view(z_q.size(0), self.config['latent_dim'], 8, 4).permute(1,0,2,3)
        z_q_reshaped = z_q.view(z_q.size(0), self.config['latent_dim'], 8, 4, 1)

        recon = self.decoder(z_q_reshaped)
        return recon, z_e_proj