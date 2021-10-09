#FROM ubuntu:18.04
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

LABEL maintainer "Joao Victor da Fonseca Pinto <jodafons@lps.ufrj.br>"

ENV LC_ALL C.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV TERM screen
USER root
SHELL [ "/bin/bash", "-c" ]


RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone



RUN apt-get update && apt-get install -y texlive-full


# Install pip packages
RUN pip install --no-cache-dir \
    setuptools \
    future \
    onnxruntime \
    onnx \
    keras2onnx \
    pandas \
    sklearn \
    seaborn \
    jupyterlab \
    prettytable \
    pickle5 \
    tqdm




WORKDIR /
