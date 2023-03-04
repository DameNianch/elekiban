import pytest

import numpy as np

from elekiban.model.image.segmentation import get_simple_model


@pytest.mark.parametrize("class_num", [1, 3])
@pytest.mark.parametrize("input_channel ", [1, 3, 5])
@pytest.mark.parametrize("input_length", [16, 32])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("model_scale", [0.3, 1, 1.1])
def test_auto_select_model(class_num, input_channel, input_length, batch_size, model_scale):
    model = get_simple_model("test_input", "test_output", class_num, model_scale, [input_length, input_length], input_channel)
    prediction = model.predict(np.ones([batch_size, input_length, input_length, input_channel]))
    assert prediction["test_output"].shape == (batch_size, input_length, input_length, class_num)
