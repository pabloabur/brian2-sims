FROM jupyter/r-notebook
USER jovyan
COPY . $HOME
RUN conda env create
RUN conda init bash
RUN conda activate app
RUN python run_simulation.py -h
RUN make -h
RUN cmake -h


