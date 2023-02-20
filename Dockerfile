FROM mambaorg/micromamba:1.3.1
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
RUN $(which python) -c "import brian2"



RUN $(which pip) install -e .
RUN $(which python) -c "import brian2"
RUN $(which python) run_simulation.py -h
USER root
RUN apt install - y cmake # develop-tools
RUN conda install conda-build
RUN conda develop .
#RUN make -h
RUN cmake -h

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
