import cv2
import numpy as np

cutout = cv2.imread("outputs/pipeline_cutout.png", cv2.IMREAD_UNCHANGED)
h, w = cutout.shape[:2]

b, g, r, alpha = cv2.split(cutout)
alpha_float = alpha.astype(float) / 255.0

background = np.zeros((h, w, 3), dtype=np.uint8)
for y in range(h):
    brightness = 235 - int((y / h) * 45)
    background[y, :] = [brightness, brightness, brightness]

center_x, center_y = w // 2, int(h * 0.3)
Y, X = np.ogrid[:h, :w]
dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
vignette = 1 - (dist / max_dist) * 0.15
vignette = np.clip(vignette, 0.85, 1.0)

for c in range(3):
    background[:, :, c] = np.clip(background[:, :, c].astype(float) * vignette, 0, 255).astype(np.uint8)

ys, xs = np.where(alpha > 50)
car_bottom = int(ys.max())
car_top = int(ys.min())
car_left = int(xs.min())
car_right = int(xs.max())
car_width = car_right - car_left
car_height = car_bottom - car_top

reflection_height = int(car_height * 0.35)

car_bgr = cv2.merge([b, g, r])
car_region = car_bgr[car_top:car_bottom, car_left:car_right]
alpha_region = alpha[car_top:car_bottom, car_left:car_right]

reflection = cv2.flip(car_region, 0)
reflection_alpha = cv2.flip(alpha_region, 0)

reflection_resized = cv2.resize(reflection, (car_width, reflection_height))
reflection_alpha_resized = cv2.resize(reflection_alpha, (car_width, reflection_height))

fade = np.linspace(0.35, 0.0, reflection_height).reshape(-1, 1)
reflection_alpha_float = (reflection_alpha_resized.astype(float) / 255.0) * fade

result = background.copy()

refl_y_start = car_bottom
refl_y_end = min(h, car_bottom + reflection_height)
actual_refl_h = refl_y_end - refl_y_start

if actual_refl_h > 0:
    for c in range(3):
        bg_slice = result[refl_y_start:refl_y_end, car_left:car_right, c].astype(float)
        refl_slice = reflection_resized[:actual_refl_h, :, c].astype(float)
        a_slice = reflection_alpha_float[:actual_refl_h, :]
        result[refl_y_start:refl_y_end, car_left:car_right, c] = (
            refl_slice * a_slice + bg_slice * (1 - a_slice)
        ).astype(np.uint8)

shadow = np.zeros((h, w), dtype=np.uint8)
cv2.ellipse(shadow, ((car_left + car_right) // 2, car_bottom), (int(car_width * 0.42), 12), 0, 0, 360, 100, -1)
shadow = cv2.GaussianBlur(shadow, (41, 41), 0)
shadow_norm = shadow.astype(float) / 255.0

for c in range(3):
    result[:, :, c] = np.clip(result[:, :, c].astype(float) - shadow_norm * 40, 0, 255).astype(np.uint8)

for c, channel in enumerate([b, g, r]):
    result[:, :, c] = (channel * alpha_float + result[:, :, c] * (1 - alpha_float)).astype(np.uint8)

cv2.imwrite("outputs/final_studio_image.jpg", result)
print("Saved: outputs/final_studio_image.jpg")