import torch
from diffusers import DiTPipeline, DDIMScheduler

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
pipe.to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)  # or DPMSolver for fewer steps

# Class-conditional example (ImageNet label)
image = pipe(
    class_labels=[206],  # e.g., golden retriever
    num_inference_steps=50,
    guidance_scale=1.5,  # CFG if desired
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]
image.save("dit_sample2.png")