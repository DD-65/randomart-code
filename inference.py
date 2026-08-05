from pathlib import Path

import torch
from diffusers import DDPMPipeline

# use MPS on Mac and fall back to the CPU on other machines
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# the model repo is next to the code repo
project_directory = Path(__file__).resolve().parent
model_directory = project_directory.parent / "randomart"
output_directory = project_directory / "generated"
output_directory.mkdir(exist_ok=True)

# load the model once and then generate multiple images with it
pipe = DDPMPipeline.from_pretrained(model_directory).to(device)
for i in range(3):
    # more denoising steps take longer but can give a bit more detail
    image = pipe(num_inference_steps=500).images[0]
    image.save(output_directory / f"generated-500_{i + 1}.png")
