FROM jupyter/r-notebook
USER jovyan
COPY . $HOME
RUN conda env create
RUN conda activate app
RUN $(which python3) run_simulation.py -h
RUN make -h
RUN cmake -h


