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
    return x / 255


os.makedirs("output/segmentation", exist_ok=True)
input_image_paths = sorted(glob("dataset/images/segmentation/*.png"))
input_pipeline = elekiban.pipeline.pipe.ImagePipe("image_input", input_image_paths, adjust_fn=div_255)

elekiban.executor.predict("output/model.h5", input_pipeline, save_as_img)
