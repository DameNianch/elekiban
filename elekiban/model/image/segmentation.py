from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, UpSampling2D, Softmax


def get_simple_model(input_name, output_name, class_num, model_scale=1, image_size=[16, 16], input_channel=3):
    filters = input_channel
    x = Input(shape=(None, None, input_channel))
    y = x
    y_bypass = []

    def calc_pooling_num(length):
        num = 0
        while True:
            if length % (2**(num + 1)) == 0:
                num += 1
            else:
                return num

    for _ in range(min([calc_pooling_num(length) for length in image_size])):
        filters = max(int(model_scale * filters * 2), class_num)
        y = Conv2D(filters, 3, padding="same", activation="relu")(y)
        y_bypass.append(y)
        y = MaxPool2D(padding="same")(y)

    y = Conv2D(filters, 3, padding="same", activation="relu")(y)
    y = Conv2D(filters, 3, padding="same", activation="relu")(y)

    for y_skip in y_bypass[::-1]:
        y = UpSampling2D()(y)
        y = Concatenate()([y_skip, y])
        filters = max(int(model_scale * filters / 2), class_num)
        y = Conv2D(filters, 3, padding="same", activation="relu")(y)

    y = Conv2D(class_num, 1, padding="same", activation="sigmoid" if class_num == 1 else "softmax")(y)

    return Model(inputs={input_name: x}, outputs={output_name: y}, name="SimpleImgSegmentation")
