from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

model_id = "stabilityai/stable-diffusion-3.5-medium"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

prompt = "A cute capybara holding a sign that says 'Hello World', vibrant colors, detailed fur, cartoon style"

# Option 1: Pyramid Attention Broadcast (PAB) — usually best starting point for SD3
#pipeline.enable_pab_cache(
#    cache_interval=3,          # recompute every N steps (higher = more aggressive)
#    cache_layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],  # or subset, e.g. cross-attn heavy layers
    # You can also tune branch ratios or other params — see docs
#)

# Option 2: FasterCache (similar spirit, sometimes slightly different quality/speed)
# pipeline.enable_faster_cache(...)          # syntax may vary — check current docs

# Option 3: TaylorSeer Cache (Taylor approximation style — good for larger skips)
# pipeline.enable_taylorseer_cache(
#     interval=4,                # how often to do full forward
#     order=2,                   # Taylor expansion order (1–3 usually)
# )

image = pipeline(
    prompt=prompt,
    num_inference_steps=40,
    guidance_scale=4.5,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]

image.save("capybara2.png")
