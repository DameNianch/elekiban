from glob import glob
import elekiban


def div_255(x):
    return x / 255


input_image_paths = sorted(glob("dataset/images/segmentation/*.png"))
input_pipeline = elekiban.pipeline.pipe.ImagePipe("image_input", input_image_paths, adjust_fn=div_255)

label_image_paths = sorted(glob("dataset/labels/segmentation/*.png"))
output_pipeline = elekiban.pipeline.pipe.ImagePipe("image_output", input_image_paths, adjust_fn=div_255)

train_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=4)
valid_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=4)


model = elekiban.model.image.segmentation.get_simple_model("image_input", "image_output", class_num=3, model_scale=2)
output_path = elekiban.executor.train(model, train_faucet, valid_faucet, epochs=250, output_path="output")
print(output_path)
