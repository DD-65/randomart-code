import torch
from diffusers import DDPMPipeline

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

pipe = DDPMPipeline.from_pretrained("randomart_pipeline").to(device)
for i in range(3):
    image = pipe(num_inference_steps=500).images[0]
    image.save(f"generated/generated-500_{i+1}.png")
