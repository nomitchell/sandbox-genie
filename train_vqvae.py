# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.

import numpy as np
import torch

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import CIFAR10
from vqvae import VQVAE
from frame_dataset import SequentialFrameDataset
import os

torch.set_printoptions(linewidth=160)

def save_img_tensors_as_img(img_tensors, output_dir):
    it = img_tensors.permute(1, 0, 2, 3)

    for i in range(it.shape[0]):
        img_it = it[i, :3]
        
        img_it = (img_it - img_it.min()) / (img_it.max() - img_it.min())
        
        img = to_pil_image(img_it)
        
        file_path = os.path.join(output_dir, f"image_{i}.png")
        img.save(file_path)

def main():
    # Initialize model.
    device = torch.device("cuda:0")
    use_ema = True
    model_args = {
        "in_channels": 3,
        "num_hiddens": 512,
        "num_downsampling_layers": 2,
        "num_residual_layers": 6,
        "num_residual_hiddens": 64,
        # this is supposed to be 32, but was getting incom in quant, need to fix
        "embedding_dim": 4,
        "num_embeddings": 1024,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    model = VQVAE(**model_args).to(device)

    # Initialize dataset.
    batch_size = 48
    workers = 10
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    data_root = "coinrun_frames"
    clip_length = 16
    train_dataset = SequentialFrameDataset(data_root, clip_length, transform)
    #train_data_variance = np.var(train_dataset.data / 255)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )

    # Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation
    # Learning".
    beta = 0.9

    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    lr = 1e-4
    optimizer = optim.Adam(train_params, lr=lr)
    criterion = nn.MSELoss()

    # Train model.
    epochs = 7
    eval_every = 100
    best_train_loss = float("inf")
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        total_recon_error = 0
        n_train = 0
        for (batch_idx, train_tensors) in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = train_tensors[0].to(device)
            out = model(imgs)
            # TODO test with and without regularization using train_data_variance
            #recon_error = criterion(out["x_recon"], imgs) / train_data_variance
            recon_error = criterion(out["x_recon"], imgs)
            total_recon_error += recon_error.item()
            loss = recon_error + beta * out["commitment_loss"]
            if not use_ema:
                loss += out["dictionary_loss"]

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

            if ((batch_idx + 1) % eval_every) == 0:
                print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
                total_train_loss /= n_train
                if total_train_loss < best_train_loss:
                    best_train_loss = total_train_loss

                print(f"total_train_loss: {total_train_loss}")
                print(f"best_train_loss: {best_train_loss}")
                print(f"recon_error: {total_recon_error / n_train}\n")

                total_train_loss = 0
                total_recon_error = 0
                n_train = 0

    # Generate and save reconstructions.
    model.eval()

    valid_dataset = SequentialFrameDataset("val_coinrun_frames", clip_length, transform)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=workers,
    )

    torch.save(model.state_dict(), "model.pth")

    with torch.no_grad():
        for valid_tensors in valid_loader:
            break

        save_img_tensors_as_img(valid_tensors[0], "output_imgs\\train")
        save_img_tensors_as_img(model(valid_tensors[0].to(device))["x_recon"], "output_imgs\\val")

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()