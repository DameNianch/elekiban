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
    version='0.0.2',
    url="https://www.elekiban.pub",
    author="DameNianch",
    license="Check https://github.com/DameNianch/elekiban",
    packages=find_packages(),
    install_requires=install_requires
)
