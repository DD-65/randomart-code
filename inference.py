import torch
from diffusers import DDPMPipeline

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

pipe = DDPMPipeline.from_pretrained("randomart_pipeline").to(device)
image = pipe(num_inference_steps=100).images[0]
image.save("generated.png")
