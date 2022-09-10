import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Conv1D, GlobalAveragePooling1D, MaxPool1D


def get_simple_model(input_name, output_name, class_num, model_scale=1, wave_length=32, input_channel=3):
    filters = input_channel
    x = Input(shape=(None, input_channel))
    y = x
    for _ in range(int(np.log2(wave_length)) - 1):
        filters = max(int(model_scale * filters * 2), 1)
        y = Conv1D(filters, 3, padding="same", activation="relu")(y)
        y = MaxPool1D(padding="same")(y)
    y = GlobalAveragePooling1D()(y)
    y = Dense(filters, activation="relu")(y)
    y = Dense(max((filters + class_num) / 2, 1), activation="relu")(y)
    y = Dense(class_num, activation="sigmoid" if class_num == 1 else "softmax")(y)
    return Model(inputs={input_name: x}, outputs={output_name: y}, name="wave2class")
