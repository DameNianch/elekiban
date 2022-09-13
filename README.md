# Elekiban
Library and Framework for python beginners who want to do AI. You can try solve general problems by python without a lot of knowledge and skills.

# Usage
If you can do python, just pip and modify sources in sample.
```
pip install elekiban
```
In the case of no suitable samples for your case, I recommend learning python and keras, tensorflow or pytorch. 

# Requirement
I will write...
- opencv-python
- Pillow
- tensorflow
- tensorflow_hub
- numpy

# Sample
In sample directory

```
python3 src/image_classification.py
```

or

```
python3 src/image_classification_pretrained.py
```

# Data Format
Acceptable data formats are shown below. Note that these are not optimal solutions for performance. If you aim for the optimal solution, you should immediately step up from the data format prepared by elekiban and proceed to tf.data. Regarding the problem that the corresponding data format is insufficient, we will respond if there is an issue. (I don't have enough time and motivation to actively make it.)
## Image
Accepts common image formats. If there is a format that needs to be addressed, please contact us through issue.
## Wave
Currently, only proprietary fixed-length formats are supported. When handling multiple waves at the same time like a sample, please make the length of each wave the same. It will be greatly improved in the future, but I am putting it off because I do not feel like it. If you have a format that needs support, please let us know via issue. I probably will.
```
{"first_channel": [0, 0, 1, 1, 0, 0, 1, 1], "second_channel": [1, 0, 1, -1, 0, 0, 0, 1]}
{"first_channel": [0, 0, 1, 1, 0, 0, 1, 1], "second_channel": [1, 0, 1, -1, 0, 0, 0, 1]}
```

## Music
comming soon

## Text
comming soon

## Lavel
Supports only vector values. If the element value is greater than 1, adjust it with adjust_fn.
Specific examples are given below, but please check the details by executing the sample.
- `[0, 1, 0, 0]`
- `[1, 0, 1, 1, 1]`
- `[30, 0, 13, 0]`

# License
Permission to use the model for service creation/research and technical verification.
