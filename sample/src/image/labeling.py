from glob import glob

import elekiban


def div_255(x):
    return x / 255


input_name = "image_input"
output_name = "label_output"

image_paths = sorted(glob("dataset/images/labeling/*.png"))
input_pipeline = elekiban.pipeline.pipe.ImagePipe(input_name, image_paths, adjust_fn=div_255)

output_labels = elekiban.pipeline.pump.load_csv("dataset/labels/labeling/label.csv")
output_pipeline = elekiban.pipeline.pipe.LabelPipe(output_name, output_labels)

train_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=3)
valid_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=3)

model = elekiban.model.image.labeling.get_simple_model(input_name, output_name, class_num=4, model_scale=0.6)
output_path = elekiban.executor.train(model, train_faucet, valid_faucet, epochs=400, output_path="output")
print(output_path)
