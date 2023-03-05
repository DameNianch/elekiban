import tempfile
import os
import pytest
import cv2
import numpy as np
from elekiban.pipeline.pipe import ImagePipe


class TestImagePipe:
    def test_error_case(self):
        img_paths = []
        with tempfile.TemporaryDirectory() as dname:
            with pytest.raises(FileExistsError):
                for i in range(10):
                    i_path = os.path.join(dname, f"{i}.png")
                    img_paths.append(i_path)
                _ = ImagePipe("image_pipe", img_paths)

    def test_normal_case(self):
        img_paths = []
        with tempfile.TemporaryDirectory() as dname:
            for i in range(10):
                i_path = os.path.join(dname, f"{i}.png")
                img_paths.append(i_path)
                cv2.imwrite(i_path, np.zeros([5, 5, 3], dtype=np.uint8))
            _ = ImagePipe("image_pipe", img_paths)
