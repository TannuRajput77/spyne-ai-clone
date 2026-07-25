import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import numpy as np
import cv2

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32
)
pipe = pipe.to("cpu")

prompt = ("empty modern car showroom, no cars, no vehicles, "
          "polished reflective white floor, soft studio lighting, "
          "low camera angle, minimalist background, photorealistic, empty room")
negative_prompt = "car, vehicle, truck, suv, people, text, watermark"

generated_bg = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=4,
    guidance_scale=1.5
).images[0]

generated_bg.save("outputs/diffusion_background.png")
print("Saved: outputs/diffusion_background.png")

cutout = cv2.imread("outputs/pipeline_cutout.png", cv2.IMREAD_UNCHANGED)
h, w = cutout.shape[:2]

bg_resized = generated_bg.resize((w, h), Image.LANCZOS)
bg_array = cv2.cvtColor(np.array(bg_resized), cv2.COLOR_RGB2BGR)

b, g, r, alpha = cv2.split(cutout)

# fix edge color spill
kernel = np.ones((3, 3), np.uint8)
alpha = cv2.erode(alpha, kernel, iterations=1)
alpha_float = alpha.astype(float) / 255.0

# realistic footprint shadow — follow car's actual bottom silhouette
mask = (alpha > 10).astype(np.uint8)
shadow_layer = np.zeros((h, w), dtype=np.uint8)

col_indices = np.where(mask.any(axis=0))[0]
for x in col_indices:
    col = np.where(mask[:, x] > 0)[0]
    if len(col) == 0:
        continue
    bottom_y = col.max()
    cv2.line(shadow_layer, (x, bottom_y), (x, bottom_y + 6), 255, 1)

shadow_layer = cv2.GaussianBlur(shadow_layer, (35, 15), 0)
shadow_alpha = (shadow_layer.astype(float) / 255.0) * 0.55

result = bg_array.astype(float)
for c in range(3):
    result[:, :, c] *= (1 - shadow_alpha)

for c, channel in enumerate([b, g, r]):
    result[:, :, c] = channel.astype(float) * alpha_float + result[:, :, c] * (1 - alpha_float)

result = result.astype(np.uint8)

cv2.imwrite("outputs/diffusion_final.jpg", result)
print("Saved: outputs/diffusion_final.jpg")