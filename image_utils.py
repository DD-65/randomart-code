import numpy as np
import torchvision
from PIL import Image


def show_images(images):
    """Convert a normalized image batch into a single Pillow image grid."""
    images = images * 0.5 + 0.5
    grid = torchvision.utils.make_grid(images)
    grid_array = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    return Image.fromarray(np.asarray(grid_array, dtype=np.uint8))


def make_grid(images, size=64):
    """Resize and arrange Pillow images in a horizontal grid."""
    output_image = Image.new("RGB", (size * len(images), size))
    for index, image in enumerate(images):
        output_image.paste(image.resize((size, size)), (index * size, 0))
    return output_image
