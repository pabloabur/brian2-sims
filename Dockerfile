FROM condaforge/mambaforge as conda
#RUN /bin/bash -c "conda init bash"
COPY . .
#RUN --mount=type=cache,target=/opt/conda/pkgs mamba create --copy -p /env --file conda-linux-64.lock
#RUN conda run -p /env python -m pip install --no-deps /pkg
RUN conda install conda-build
RUN conda develop .
#FROM jupyter/r-notebook
#COPY --from=conda /env /env

RUN python run_simulation.py -h
RUN make -h
RUN cmake -h


## Now add any local files from your repository.
## As an example, we add a Python package into
## the environment.
COPY . /pkg
RUN conda run -p /env python -m pip install --no-deps /pkg

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
