import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
import cv2

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cpu")

prompt = ("realistic car dealership showroom interior, neutral grey and white tones, "
          "polished concrete floor with soft reflection, large glass windows, "
          "soft diffused studio lighting, minimal modern architecture, empty showroom, "
          "professional automotive photography, subtle ambient lighting, "
          "clean and elegant, photorealistic, 8k, "
          "muted color palette, cool white balance, architectural photography, "
          "minimalist showroom design, soft shadows, high-end dealership")

negative_prompt = ("colorful, saturated colors, neon lights, cartoon, illustration, "
                    "cluttered, people, text, logo, watermark, low quality, blurry, "
                    "dramatic lighting, warm orange tones, busy background, outdoor scene, "
                    "car, vehicle, truck, suv")

generated_bg = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]

generated_bg.save("outputs/raw_diffusion_background.png")
print("Saved: outputs/raw_diffusion_background.png")

w, h = generated_bg.size
crop_top = int(h * 0.30)
generated_bg = generated_bg.crop((0, crop_top, w, h))

cutout = cv2.imread("outputs/raw_test_cutout.png", cv2.IMREAD_UNCHANGED)
ch, cw = cutout.shape[:2]

bg_resized = generated_bg.resize((cw, ch), Image.LANCZOS)
bg_array = cv2.cvtColor(np.array(bg_resized), cv2.COLOR_RGB2BGR)

b, g, r, alpha = cv2.split(cutout)
kernel = np.ones((3, 3), np.uint8)
alpha = cv2.erode(alpha, kernel, iterations=1)
alpha_float = alpha.astype(float) / 255.0

car_bgr = cv2.merge([b, g, r])
car_lab = cv2.cvtColor(car_bgr, cv2.COLOR_BGR2LAB).astype(float)
bg_lab = cv2.cvtColor(bg_array, cv2.COLOR_BGR2LAB).astype(float)

mask_bool = alpha > 10
car_l_mean = car_lab[:, :, 0][mask_bool].mean()
bg_l_mean = bg_lab[:, :, 0].mean()
l_shift = (bg_l_mean - car_l_mean) * 0.30

car_lab[:, :, 0] = np.clip(car_lab[:, :, 0] + l_shift, 0, 255)
car_matched = cv2.cvtColor(car_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
b, g, r = cv2.split(car_matched)

mask_u8 = (alpha > 10).astype(np.uint8)
shadow_layer = np.zeros((ch, cw), dtype=np.uint8)
col_indices = np.where(mask_u8.any(axis=0))[0]
for x in col_indices:
    col = np.where(mask_u8[:, x] > 0)[0]
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
cv2.imwrite("outputs/raw_test_diffusion_final.jpg", result)
print("Saved: outputs/raw_test_diffusion_final.jpg")