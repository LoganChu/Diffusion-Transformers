import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16, 
)

pipe.to("cuda")

# Optional: enable CPU offload if VRAM is tight (trades speed for lower memory)
# pipe.enable_model_cpu_offload()

prompt = "A cute capybara holding a sign that says 'Hello World', vibrant colors, detailed fur, cartoon style"

image = pipe(
    prompt=prompt,
    negative_prompt="blurry, low quality, distorted, ugly",  # optional
    height=1024,
    width=1024,                          # or start with 768×768 to save VRAM
    num_inference_steps=28,              # 20–50 is typical; 28 is a good balance
    guidance_scale=5.0,                  # 4.5–7.0 common for SD 3.5
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("sd35_medium_example.png")
image.show() 