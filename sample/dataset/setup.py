import os
import csv
from PIL import Image, ImageChops, ImageDraw

os.makedirs("dataset/images/vectorization", exist_ok=True)
os.makedirs("dataset/labels/vectorization", exist_ok=True)
labels = []
img_size = (16, 16)
for i in range(20):
    i_color = (i * 10, 255 - i * 10, abs(255 - i * 25))
    img = Image.new("RGB", img_size, i_color)
    img.save(f"dataset/images/vectorization/{i:04}.png")
    labels.append(i_color)

with open("dataset/labels/vectorization/label.csv", "w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(labels)


os.makedirs("dataset/images/classification", exist_ok=True)
os.makedirs("dataset/labels/classification", exist_ok=True)
labels = []
img_size = (64, 64)
for i in range(10):
    i_color = (i * 10, 255 - i * 10, abs(255 - i * 25))
    img = Image.new("RGB", img_size, i_color)
    draw = ImageDraw.Draw(img)
    if i % 2 == 0:
        draw.pieslice((4, 4, 60, 60), start=i, end=270 + i, fill=(0, 0, 0))
        labels.append([0])
    else:
        draw.rectangle((8, 8, 56, 56), fill=(255, 255, 255))
        labels.append([1])
    img.save(f"dataset/images/classification/{i:04}.png")
