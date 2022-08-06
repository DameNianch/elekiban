from glob import glob

from PIL import Image
import numpy as np

import elekiban


def resize_and_div255(x):
    img = Image.fromarray(x)
    img = img.resize([224, 224])
    return np.array(img) / 255


image_paths = sorted(glob("dataset/images/classification/*.png"))
input_pipeline = elekiban.pipeline.pipe.ImagePipe("image_input", image_paths, adjust_fn=resize_and_div255)

output_labels = elekiban.pipeline.pump.load_csv("dataset/labels/classification/label.csv")
output_pipeline = elekiban.pipeline.pipe.LabelPipe("label_output", output_labels)

train_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=6)
valid_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=6)

model = elekiban.model.image.classification.get_efficient_net_v2("image_input", "label_output", class_num=1)
output_path = elekiban.executor.train(model, train_faucet, valid_faucet, epochs=10, output_path="output")
print(output_path)
