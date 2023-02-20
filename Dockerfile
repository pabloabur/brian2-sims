FROM condaforge/mambaforge
#RUN mamba update -n base mamba conda
COPY . $HOME
RUN mamba env create --file environment.yml
RUN conda env create
FROM jupyter/r-notebook

RUN python run_simulation.py -h
RUN make -h
RUN cmake -h

#RUN mamba install -n rnaquant --revision 1
#mamba install -n rnaquant fastqc

#RUN mamba activate app

#USER jovyan
# RUN /bin/bash -c "mamba app create -f environment.yml"
#RUN conda env create
#SHELL ["conda", "init", "bash", "-c"]
#RUN conda init bash
#RUN echo "conda activate app" >> ~/.bashrc
#SHELL ["/bin/bash", "--login", "-c"]
#SHELL ["conda", "run", "-n", "app", "/bin/bash", "-c"]
