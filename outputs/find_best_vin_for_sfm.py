import os
from collections import defaultdict

SOURCE = "D:/dataset"

vin_counts = {}
for vin_folder in os.listdir(SOURCE):
    folder_path = os.path.join(SOURCE, vin_folder)
    if not os.path.isdir(folder_path):
        continue
    count = len([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    vin_counts[vin_folder] = count

sorted_vins = sorted(vin_counts.items(), key=lambda x: x[1], reverse=True)

print("Top 10 VINs by image count:")
for vin, count in sorted_vins[:10]:
    print(f"{vin}: {count} images")