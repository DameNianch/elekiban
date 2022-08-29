import os
import csv
import random
import numpy as np


wave_dir = "dataset/wave/classification/fixed_length"
label_dir = "dataset/wave/classification/fixed_length"
os.makedirs(wave_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
waves = []
labels = []
wave_length = 31
data_num = 40
for i in range(data_num):
    i_wave = {}
    phi = random.uniform(np.pi / 2, 1.5 * np.pi)
    ticks = [2 * np.pi * i / wave_length + phi for i in range(wave_length)]
    i_wave["a"] = np.round(np.sin(ticks), 4).tolist()
    i_wave["b"] = np.round(np.cos(ticks) if i % 2 == 0 else np.sin(ticks), 3).tolist()
    waves.append(str(i_wave))
    labels.append([0 if i % 2 == 0 else 1])

with open(os.path.join(wave_dir, "wave.jsonl"), "w") as f:
    f.write("\n".join(waves))

with open(os.path.join(label_dir, "label.csv"), "w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(labels)
