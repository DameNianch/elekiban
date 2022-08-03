from PIL import Image, ImageChops
import csv

labels = []
img_size = (16, 16)
for i in range(100):
    i_color = (i * 2, 255 - i * 2, abs(255 - i * 5))
    img = Image.new("RGB", img_size, i_color)
    img = ImageChops.multiply(img, Image.effect_noise(img_size, 100).convert('RGB'))
    img.save(f"dataset/images/{i}.png")
    labels.append(i_color)

with open("dataset/labels/label.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(labels)
