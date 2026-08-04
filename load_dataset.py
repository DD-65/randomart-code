import torch
from datasets import load_dataset
from diffusers import DDPMScheduler
from torchvision import transforms

# Or load images from a local folder
dataset = load_dataset(
    "imagefolder",
    data_dir="wikiart_under_2400_resized_dataset",
    split="train",
)

# resize images to 256x256 / 128x128
image_size = 128
# You can lower your batch size if you're running out of GPU memory
batch_size = 24


# Define data augmentations. Random cropping preserves the original aspect ratio.
# We also want to flip the images horizontally to get more variety in the training data
preprocess = transforms.Compose(
    [
        transforms.Resize(
            image_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# ------------- preprocessing
def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)

# Create a dataloader from the dataset to serve up the transformed images in batches
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

# ----------- noising
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
