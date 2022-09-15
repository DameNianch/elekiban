import os
import csv
import random
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

os.makedirs("dataset/images/labeling", exist_ok=True)
os.makedirs("dataset/labels/labeling", exist_ok=True)
labels = []
img_size = (16, 16)
for i in range(10):
    i_label = [random.randint(0, 1) for _ in range(3)]
    img = Image.new("RGB", img_size, tuple([255 * c for c in i_label]))
    draw = ImageDraw.Draw(img)
    if i % 2 == 0:
        i_label.append(1)
        draw.rectangle((4, 4, 8, 8), fill=(255, 255, 255))
    else:
        i_label.append(0)
    img.save(f"dataset/images/labeling/{i:04}.png")
    labels.append(i_label)

with open("dataset/labels/labeling/label.csv", "w") as f:
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

with open("dataset/labels/classification/label.csv", "w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(labels)

os.makedirs("dataset/images/segmentation", exist_ok=True)
os.makedirs("dataset/labels/segmentation", exist_ok=True)
img_size = (64, 64)
for i in range(10):
    i_color = (i * 20, 255 - i * 20, abs(255 - i * 50))
    img_in = Image.new("RGB", img_size, i_color)
    draw_in = ImageDraw.Draw(img_in)
    draw_in.pieslice((4, 4, 60, 60), start=i * 20, end=270 + i * 20, fill=(0, 0, 0))
    draw_in.rectangle((20, 20, 44, 44), fill=(255, 255, 255))
    img_in.save(f"dataset/images/segmentation/{i:04}.png")
    img_out = Image.new("RGB", img_size, (255, 0, 0))
    draw_out = ImageDraw.Draw(img_out)
    draw_out.pieslice((4, 4, 60, 60), start=i * 20, end=270 + i * 20, fill=(0, 255, 0))
    draw_out.rectangle((20, 20, 44, 44), fill=(0, 0, 255))
    img_out.save(f"dataset/labels/segmentation/{i:04}.png")
