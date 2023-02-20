FROM condaforge/mambaforge as conda
ARG USERNAME=jovyan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.

COPY . .
RUN apt install cmake # develop-tools
RUN chown -r . $USERNAME
USER $USERNAME

RUN conda install conda-build
RUN conda develop .
#RUN make -h
RUN cmake -h

RUN $(which pip) install -e .

RUN $(which python) run_simulation.py -h
RUN $(which python) -c "import brian2"
RUN $(which python) run_simulation.py -h



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
