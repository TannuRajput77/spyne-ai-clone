import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(scale=4, model_path="models/RealESRGAN_x4plus.pth", model=model, tile=200, half=False)

img = cv2.imread("outputs/raw_test_cutout.png", cv2.IMREAD_UNCHANGED)
output, _ = upsampler.enhance(img, outscale=4)
cv2.imwrite("outputs/raw_test_enhanced.png", output)
print("Saved: outputs/raw_test_enhanced.png")
print(f"Original: {img.shape} -> Enhanced: {output.shape}")