from setuptools import setup, find_packages

install_requires = [
    "opencv-python",
    "Pillow",
    "tensorflow",
    "tensorflow_hub",
    "numpy"
]

setup(
    name='elekiban',
    version='0.0.0',
    packages=find_packages(),
    install_requires=install_requires
)
