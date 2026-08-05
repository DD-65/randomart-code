from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMPipeline

from load_dataset import noise_scheduler, train_dataloader


# main training settings and output folders
NUM_EPOCHS = 20
CHECKPOINT_INTERVAL = 5
PROJECT_DIRECTORY = Path(__file__).resolve().parent
CHECKPOINT_DIRECTORY = PROJECT_DIRECTORY / "checkpoints"
MODEL_DIRECTORY = PROJECT_DIRECTORY.parent / "randomart"


def build_model():
    # the UNet gets a noisy RGB image and tries to predict its noise
    return UNet2DModel(
        sample_size=128,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        # later blocks use more channels to learn more complex features
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
    # epoch numbers are padded, so sorting also gives the newest checkpoint
    checkpoint_files = sorted(CHECKPOINT_DIRECTORY.glob("epoch-*/training_state.pt"))
    return checkpoint_files[-1] if checkpoint_files else None


def save_checkpoint(model, optimizer, losses, completed_epoch):
    checkpoint_directory = CHECKPOINT_DIRECTORY / f"epoch-{completed_epoch:03d}"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    # save a pipeline that can already be used to generate images
    image_pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    image_pipeline.save_pretrained(checkpoint_directory / "pipeline")

    # optimizer state and losses are needed to continue training later
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
        # no checkpoint means training starts from epoch zero
        return 0, []

    # load both the model weights and the old optimizer state
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
    # use the Mac GPU when it is available
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

            # every image gets a random point from the noising process
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=clean_images.device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_prediction = model(noisy_images, timesteps, return_dict=False)[0]

            # compare predicted noise with the noise that was actually added
            loss = F.mse_loss(noise_prediction, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            epoch_loss = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Epoch: {epoch + 1}, loss: {epoch_loss}")
            save_checkpoint(model, optimizer, losses, epoch + 1)

    # save the finished model in the separate Hugging Face model folder
    image_pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    image_pipeline.save_pretrained(MODEL_DIRECTORY)


if __name__ == "__main__":
    main()
