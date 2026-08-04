from pathlib import Path
from PIL import Image as PILImage
from datasets import load_dataset

output_dir = Path("wikiart_under_2400_resized_dataset")
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "huggan/wikiart",
    split="train",
    streaming=True,
)

saved = 0

for example in dataset:
    image = example["image"]

    if image.width >= 2400:
        continue

    image = image.convert("RGB")
    image = image.resize(
        (256, 256),
        resample=PILImage.Resampling.LANCZOS,
    )

    image.save(
        output_dir / f"{saved:06d}.jpg",
        format="JPEG",
        quality=90,
        optimize=True,
    )

    saved += 1
    print(f"Saved: {saved} images", end="\r")

print(f"Saved {saved} images to {output_dir}")