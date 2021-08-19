FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --fix-missing
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu==2.4.0
RUN pip3 install torch==1.7.0
RUN pip3 install tensorflow-datasets

# install openvino
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTP_PROXY
ARG DOWNLOAD_LINK=https://registrationcenter-download.intel.com/akdlm/irc_nas/17662/l_openvino_toolkit_p_2021.3.394.tgz
ARG INSTALL_DIR=/opt/intel/computer_vision_sdk
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
RUN pip3 install networkx
RUN pip3 install defusedxml

# install requirements for yolov5
COPY yolov5/requirements.txt /yolov5/requirements.txt
RUN cd /yolov5 && pip3 install -r requirements.txt
RUN apt-get update --fix-missing
RUN apt-get install -y libgl1 libgtk-3-0 libgtk-3-dev
RUN pip3 install pycocotools

# install requirements for model conversion
RUN pip3 install netron
RUN pip3 install onnx
RUN pip3 install onnx-simplifier
RUN pip3 install tensorflow-datasets
RUN pip3 install openvino2tensorflow==1.17.2
RUN pip3 install gdown