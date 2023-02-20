FROM mambaorg/micromamba:1.3.1
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -c 'import uuid; print(uuid.uuid4())' > /tmp/my_uuid
RUN python -c "import brian2"
COPY . .
USER $MAMBA_USER
RUN python run_simulation.py -h

#USER root
#RUN apt install - y cmake # develop-tools
#RUN cmake -h
#RUN conda install conda-build
#RUN conda develop .
#RUN make -h
#FROM condaforge/mambaforge as conda
#COPY . .
#USER mambauser
#WORKDIR mambauser

## Now add any local files from your repository.
## As an example, we add a Python package into
## the environment.
#COPY . /pkg
#RUN conda run -p /env python -m pip install --no-deps /pkg

# Distroless for execution
#FROM gcr.io/distroless/base-debian10

## Copy over the conda environment from the previous stage.
## This must be located at the same location.


#COPY . $HOME
#RUN mamba env create --file environment.yml
#RUN conda activate env
#RUN mamba install -n rnaquant --revision 1
#mamba install -n rnaquant fastqc
#RUN mamba update -n base mamba conda

#RUN mamba activate app

#USER jovyan
# RUN /bin/bash -c "mamba app create -f environment.yml"
#RUN conda env create
#SHELL ["conda", "init", "bash", "-c"]
#RUN conda init bash
#RUN echo "conda activate app" >> ~/.bashrc
#SHELL ["/bin/bash", "--login", "-c"]
#SHELL ["conda", "run", "-n", "app", "/bin/bash", "-c"]
