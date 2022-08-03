from glob import glob

from PIL import Image
import numpy as np
import tensorflow as tf

import elekiban


def div_255(x):
    return x / 255


def resize_and_div255(x):
    img = Image.fromarray(x)
    img = img.resize([224, 224])
    return np.array(img) / 255


image_paths = glob("dataset/images/*.png")
input_pipeline = elekiban.pipeline.pipe.ImagePipe("image_input", image_paths, adjust_fn=div_255)

output_labels = elekiban.pipeline.pump.load_csv("dataset/labels/label.csv")
output_pipeline = elekiban.pipeline.pipe.LabelPipe("label_output", output_labels, adjust_fn=div_255)

train_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=10)
valid_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=10)


model = elekiban.model.image.vectorization.get_efficient_net_v2("image_input", "label_output", 3)
output_path = elekiban.executor.train(model, train_faucet, valid_faucet, epochs=5, output_path="output")
print(output_path)
