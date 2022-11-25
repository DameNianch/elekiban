import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, MaxPool2D, Resizing
import tensorflow_hub as hub


def get_simple_model(input_name, output_name, class_num, model_scale=1, image_size=[16, 16], input_channel=3):
    filters = input_channel
    x = Input(shape=(None, None, input_channel))
    y = x
    for _ in range(int(np.log2(min(image_size))) - 1):
        filters = max(int(model_scale * filters * 2), 1)
        y = Conv2D(filters, 3, padding="same", activation="relu")(y)
        y = MaxPool2D(padding="same")(y)
    y = GlobalAveragePooling2D()(y)
    y = Dense(filters, activation="relu")(y)
    y = Dense(max((filters + class_num) / 2, 1), activation="relu")(y)
    y = Dense(class_num, activation="sigmoid" if class_num == 1 else "softmax")(y)
    return Model(inputs={input_name: x}, outputs={output_name: y}, name="img2class")


def get_efficient_net_v2(input_name, output_name, class_num, input_channel=3):
    x = Input(shape=(224, 224, input_channel))
    y = Conv2D(3, 1, padding="same")(x) if input_channel != 3 else x
    y = hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2", trainable=True)(x)
    y = Dense(640, activation="relu")(y)
    y = Dense(int((640 + class_num) / 2), activation="relu")(y)
    y = Dense(class_num, activation="sigmoid" if class_num == 1 else "softmax")(y)
    return Model(inputs={input_name: x}, outputs={output_name: y}, name="img2class")


def auto_select_model(input_name, output_name, class_num, input_channel=3, input_height=224, input_width=224):
    length_keys = [224, 260, 300, 384, 480, 512]
    model_table = {
        224: "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2",
        260: "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2",
        300: "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
        384: "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2",
        # 480: "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
        480: "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/feature_vector/2",
        512: "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2",
    }
    characteristic_length = (input_height + input_width) / 2
    target_length = length_keys[np.abs(np.array(length_keys) - characteristic_length).argmin()]
    model_path = model_table[target_length]

    x = Input(shape=(input_height, input_width, input_channel))
    y = Conv2D(3, 1, padding="same")(x) if input_channel != 3 else x
    y = Resizing(target_length, target_length)(y)

    y = hub.KerasLayer(model_path, trainable=True)(y)
    y = Dense(640, activation="relu")(y)
    y = Dense(int((640 + class_num) / 2), activation="relu")(y)
    y = Dense(class_num, activation="sigmoid" if class_num == 1 else "softmax")(y)
    return Model(inputs={input_name: x}, outputs={output_name: y}, name="img2class")
