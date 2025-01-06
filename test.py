import torch
from torchvision.transforms.functional import to_pil_image
import os

output_dir = "output_imgs"

it = torch.load("img_tensors.pt")

it = it.permute(1, 0, 2, 3)

print(it)
print(it.shape)

for i in range(it.shape[0]):
    img_it = it[i, :3]
    
    img_it = (img_it - img_it.min()) / (img_it.max() - img_it.min())
    
    img = to_pil_image(img_it)
    
    file_path = os.path.join(output_dir, f"image_{i}.png")
    img.save(file_path)
    print(f"Saved: {file_path}")

print("All images saved")