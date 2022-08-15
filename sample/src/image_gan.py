from glob import glob
import numpy as np
import elekiban

INPUT_DIM = 4


def random_fn(index):
    del index
    return np.random.normal(0, 1, INPUT_DIM)


def const_fn(index):
    del index
    return np.arange(INPUT_DIM)


random_pipe = elekiban.pipeline.pipe.CustomPipe("random_in", random_fn)
const_pipe = elekiban.pipeline.pipe.CustomPipe("const_in", const_fn)
mixed_pipe = elekiban.pipeline.pipe.MixedPipe("mixed_pipe", [random_pipe, const_pipe], [7, 2])

train_faucet = elekiban.pipeline.toolbox.SimpleFaucet([mixed_pipe], [mixed_pipe], batch_size=6)

for i in train_faucet.turn_on():
    print(i)


def div_255(x):
    return x / 255


image_paths = sorted(glob("dataset/images/classification/*.png"))
input_pipeline = elekiban.pipeline.pipe.ImagePipe("image_input", image_paths, adjust_fn=div_255)

output_labels = elekiban.pipeline.pump.load_csv("dataset/labels/classification/label.csv")
output_pipeline = elekiban.pipeline.pipe.LabelPipe("label_output", output_labels)

train_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=3)
valid_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=3)

model = elekiban.model.image.classification.get_simple_model("image_input", "label_output", class_num=1, model_scale=0.6)
output_path = elekiban.executor.train(model, train_faucet, valid_faucet, epochs=200, output_path="output")
print(output_path)
