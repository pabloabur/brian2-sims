#FROM jupyter/r-notebook
#USER jovyan
#COPY . $HOME
#WORKDIR $HOME

#RUN conda env create && echo "conda activate app" >> ~/.profile
#RUN $(which python3) run_simulation.py -h
#RUN make -h
#RUN cmake -h

# Based on answers from post below:
# https://stackoverflow.com/questions/54437030/how-can-i-create-a-docker-image-to-run-both-python-and-r
FROM ubuntu:latest
SHELL [ "/bin/bash", "--login", "-c" ]
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV DEBIAN_FRONTEND noninteractive
ARG CONDA_HASH=32d73e1bc33fda089d7cd9ef4c1be542616bd8e437d1f77afeeaf7afdb019787

WORKDIR /app
RUN apt update && apt install -y wget
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh \
    && sha256sum Miniconda3-py310_23.1.0-1-Linux-x86_64.sh \
    | grep $CONDA_HASH; if [ $? != 0 ]; then exit 1; fi \
    && mkdir /root/.conda && bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py310_23.1.0-1-Linux-x86_64.sh \
    && conda update -y conda && conda init bash
RUN apt install -y r-base

COPY . /app
RUN conda env create && echo "conda activate app" >> ~/.profile
RUN Rscript requirements.R
