from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMPipeline

from load_dataset import noise_scheduler, train_dataloader


NUM_EPOCHS = 20
CHECKPOINT_INTERVAL = 5
CHECKPOINT_DIRECTORY = Path("checkpoints")


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


def find_latest_checkpoint():
    checkpoint_files = sorted(CHECKPOINT_DIRECTORY.glob("epoch-*/training_state.pt"))
    return checkpoint_files[-1] if checkpoint_files else None


def save_checkpoint(model, optimizer, losses, completed_epoch):
    checkpoint_directory = CHECKPOINT_DIRECTORY / f"epoch-{completed_epoch:03d}"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    # This snapshot can be loaded directly for image generation.
    image_pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    image_pipeline.save_pretrained(checkpoint_directory / "pipeline")

    # This state is what allows training to resume with the same optimizer.
    torch.save(
        {
            "completed_epoch": completed_epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
        },
        checkpoint_directory / "training_state.pt",
    )
    print(f"Saved checkpoint to {checkpoint_directory}")


def restore_latest_checkpoint(model, optimizer):
    checkpoint_file = find_latest_checkpoint()
    if checkpoint_file is None:
        return 0, []

    checkpoint_directory = checkpoint_file.parent
    saved_pipeline = DDPMPipeline.from_pretrained(checkpoint_directory / "pipeline")
    model.load_state_dict(saved_pipeline.unet.state_dict())

    training_state = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(training_state["optimizer_state_dict"])
    completed_epoch = training_state["completed_epoch"]
    losses = training_state["losses"]
    print(f"Resuming from {checkpoint_directory} after epoch {completed_epoch}")
    return completed_epoch, losses


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    start_epoch, losses = restore_latest_checkpoint(model, optimizer)

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"Epoch {epoch + 1} / {NUM_EPOCHS}")
        for batch_number, batch in enumerate(train_dataloader, start=1):
            print(f"Batch {batch_number} / {len(train_dataloader)}")
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

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            epoch_loss = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Epoch: {epoch + 1}, loss: {epoch_loss}")
            save_checkpoint(model, optimizer, losses, epoch + 1)

    image_pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    image_pipeline.save_pretrained("randomart_pipeline")


if __name__ == "__main__":
    main()
