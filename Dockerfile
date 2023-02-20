FROM condaforge/mambaforge
COPY . $HOME
RUN mamba app create -f environment.yml
# RUN /bin/bash -c "mamba app create -f environment.yml"
RUN mamba activate app

RUN python run_simulation.py -h
RUN make -h
RUN cmake -h


#FROM jupyter/r-notebook

#USER jovyan

#RUN conda env create
#SHELL ["conda", "init", "bash", "-c"]
#RUN conda init bash
#RUN echo "conda activate app" >> ~/.bashrc
#SHELL ["/bin/bash", "--login", "-c"]
#SHELL ["conda", "run", "-n", "app", "/bin/bash", "-c"]
