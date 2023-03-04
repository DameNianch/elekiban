import os
from glob import glob
import cv2
import numpy as np
import elekiban


def save_as_img(x, i):
    x = x["image_output"][0]
    x = 255 * np.identity(3)[np.reshape(np.argmax(x, axis=-1), [-1])].reshape(x.shape)
    cv2.imwrite(f"output/segmentation/{i}.png", np.uint8(x))


def div_255(x):
    # x = np.flip(x, axis=1)  # If agmentation is not used, the output will be buggy.
    return x / 255


os.makedirs("output/segmentation", exist_ok=True)
input_image_paths = sorted(glob("dataset/images/segmentation/*.png"))
input_pipeline = elekiban.pipeline.pipe.ImagePipe("image_input", input_image_paths, adjust_fn=div_255)

# MEMO: loss should be less than 3e-2
elekiban.executor.predict("output/model.h5", input_pipeline, save_as_img)
