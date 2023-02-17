FROM tensorflow/tensorflow:2.10.1-gpu

RUN apt-get update && apt-get install -y git nano
RUN apt-get install -y --no-install-recommends python3-pip libgl1-mesa-dev libopencv-dev

RUN pip install --upgrade pip
RUN pip install black flake8 pytest

ENV TF_FORCE_GPU_ALLOW_GROWTH true

RUN adduser zeus
