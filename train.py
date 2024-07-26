import torch, os
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import torchvision
from accelerate import Accelerator

from dataset import ChestXray
from diffusion_model import DiffusionModel
from interpolate import Transform
from config import Datatypes

def train(
        train_size: int = 10000,
        datatype: Datatypes = Datatypes.FLOAT16,
        unet_epochs: int = 100,
        unet_output_dir: str = 'training_unet',
        vae_epochs: int = 100,
        vae_output_dir = 'training_vae',
        ):
    dm = DiffusionModel(datatype=datatype)
    vae = dm.vae
    unet = dm.unet
    scheduler = dm.scheduler

    transform = Transform()
    ds = ChestXray(dm, transform=transform, indices=[0, 2, 5])
    ds = Subset(ds, list(range(train_size)))
    dataset = DataLoader(ds, batch_size=2, shuffle=True)

    unet.train()
    optimizer = optim.AdamW(unet.parameters(), lr=1e-4)

    os.makedirs(unet_output_dir, exist_ok=True)
    unet_log_file = os.path.join(unet_output_dir, 'training_log.txt')

    accelerator = Accelerator()
    unet, optimizer, dataset = accelerator.prepare(unet, optimizer, dataset)

    running_loss = 0.0
    print_interval = 10
    for epoch in range(unet_epochs):
        for batch_idx, (_, images, hidden_states) in enumerate(dataset):
            noisy_images = torch.rand_like(images, dtype=datatype.to_torch_dtype())
            
            for timestep in scheduler.timesteps:
                current_images = scheduler.add_noise(images, noisy_images, timestep)
                current_images = torch.cat([current_images, torch.randn_like(current_images[:, :1, :, :])], dim=1)
                
                pred_images = unet(current_images, timestep, hidden_states).sample
                act_images = scheduler.add_noise(images, noisy_images, timestep - timestep)
                act_images = torch.cat([act_images, torch.randn_like(act_images[:, :1, :, :])], dim=1)
                
                loss = torchvision.ops.sigmoid_focal_loss(pred_images, act_images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                if (batch_idx + 1) % print_interval == 0:
                    print(f'Epoch [{epoch + 1}/{unet_epochs}], Batch [{batch_idx + 1}], Loss: {running_loss / print_interval:.4f}')
                    running_loss = 0.0

                with open(unet_log_file, 'a') as f:
                    f.write(f'Epoch [{epoch + 1}/{unet_epochs}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}\n')

        epoch_model_path = os.path.join(unet_output_dir, f'unet_state_dict_epoch_{epoch + 1}.pth')
        torch.save(unet.state_dict(), epoch_model_path)
        print(f'Model state dict for epoch {epoch + 1} saved to {epoch_model_path}')

    vae.train()
    optimizer = optim.AdamW(unet.parameters(), lr=1e-4)
    criterion = nn.KLDivLoss()

    os.makedirs(vae_output_dir, exist_ok=True)
    vae_log_file = os.path.join(vae_output_dir, 'training_log.txt')

    accelerator = Accelerator()
    vae, optimizer = accelerator.prepare(vae, optimizer)

    running_loss = 0.0
    print_interval = 10
    for epoch in range(vae_epochs):
        running_loss = 0.0
        for i, (latents, _, _) in enumerate(dataset):
            pixel_values = latents.to(device)
            reconstructions = vae(pixel_values).sample
            loss = criterion(reconstructions, pixel_values)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if (i + 1) % print_interval == 0:
                print(f"Epoch [{epoch+1}/{vae_epochs}], Step [{i+1}/{len(dataset)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{vae_epochs}], Average Loss: {epoch_loss:.4f}")

        epoch_model_path = os.path.join(vae_output_dir, f'vae_state_dict_epoch_{epoch+1}.pth')
        torch.save(vae.state_dict(), epoch_model_path)

        with open(vae_log_file, 'a') as f:
            f.write(f"Epoch [{epoch+1}/{vae_epochs}], Average Loss: {epoch_loss:.4f}\n")


if __name__ == "__main__":
    train()
