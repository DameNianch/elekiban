from glob import glob

from PIL import Image
import numpy as np

import elekiban


def resize_and_div255(x):
    img = Image.fromarray(x)
    img = img.resize([224, 224])
    return np.array(img) / 255


input_name = "image_input"
output_name = "label_output"

image_paths = sorted(glob("dataset/images/labeling/*.png"))
input_pipeline = elekiban.pipeline.pipe.ImagePipe(input_name, image_paths, adjust_fn=resize_and_div255)

output_labels = elekiban.pipeline.pump.load_csv("dataset/labels/labeling/label.csv")
output_pipeline = elekiban.pipeline.pipe.LabelPipe(output_name, output_labels)

train_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=6)
valid_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=6)

model = elekiban.model.image.labeling.get_efficient_net_v2(input_name, output_name, class_num=4)
output_path = elekiban.executor.train(model, train_faucet, valid_faucet, epochs=20, output_path="output")
print(output_path)
