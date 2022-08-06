from glob import glob
import elekiban


def div_255(x):
    return x / 255


image_paths = sorted(glob("dataset/images/vectorization/*.png"))
input_pipeline = elekiban.pipeline.pipe.ImagePipe("image_input", image_paths, adjust_fn=div_255)

output_labels = elekiban.pipeline.pump.load_csv("dataset/labels/vectorization/label.csv")
output_pipeline = elekiban.pipeline.pipe.LabelPipe("label_output", output_labels, adjust_fn=div_255)

train_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=4)
valid_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=4)


model = elekiban.model.image.vectorization.get_simple_model("image_input", "label_output", 3, model_scale=0.1)
output_path = elekiban.executor.train(model, train_faucet, valid_faucet, epochs=5, output_path="output")
print(output_path)
