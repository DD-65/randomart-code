import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMPipeline

from load_dataset import noise_scheduler, train_dataloader


def build_model():
    return UNet2DModel(
        sample_size=128,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    losses = []

    for epoch in range(35):
        print(f"Epoch {epoch + 1} / 35")
        for batch in train_dataloader:
            print(f"Batch {len(losses) % len(train_dataloader) + 1} / {len(train_dataloader)}")
            clean_images = batch["images"].to(device)
            noise = torch.randn_like(clean_images)
            batch_size = clean_images.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=clean_images.device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_prediction = model(noisy_images, timesteps, return_dict=False)[0]

            loss = F.mse_loss(noise_prediction, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 5 == 0:
            epoch_loss = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Epoch: {epoch + 1}, loss: {epoch_loss}")

    image_pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    image_pipeline.save_pretrained("randomart_pipeline")


if __name__ == "__main__":
    main()
