from glob import glob
import numpy as np
import elekiban


def div_255(x):
    return x / 255


def flip_pair_image(inputs, outputs):
    if np.random.randint(0, 2) == 1:
        inputs["image_input"] = np.flip(inputs["image_input"], axis=2)
        outputs["image_output"] = np.flip(outputs["image_output"], axis=2)
    return inputs, outputs


input_image_paths = sorted(glob("dataset/images/segmentation/*.png"))
input_pipeline = elekiban.pipeline.pipe.ImagePipe("image_input", input_image_paths, adjust_fn=div_255)

label_image_paths = sorted(glob("dataset/labels/segmentation/*.png"))
output_pipeline = elekiban.pipeline.pipe.ImagePipe("image_output", label_image_paths, adjust_fn=div_255)

train_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=3, pairing_adjust_fn=flip_pair_image)
valid_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=3, pairing_adjust_fn=flip_pair_image)


model = elekiban.model.image.segmentation.get_simple_model("image_input", "image_output", class_num=3, model_scale=0.5, image_size=[32, 32])
output_path = elekiban.executor.train(model, train_faucet, valid_faucet, epochs=500, output_path="output")
print(output_path)
