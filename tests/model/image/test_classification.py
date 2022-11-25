import pytest

import numpy as np

from elekiban.model.image.classification import auto_select_model


@pytest.mark.parametrize("class_num", [1, 10])
@pytest.mark.parametrize("input_channel ", [1, 3, 5])
@pytest.mark.parametrize("input_length", [223, 224, 225, 384])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_auto_select_model(class_num, input_channel, input_length, batch_size):
    model = auto_select_model("test_input", "test_output", class_num, input_channel, input_length, input_length)
    prediction = model.predict(np.ones([batch_size, input_length, input_length, input_channel]))
    assert prediction["test_output"].shape == (batch_size, class_num)
