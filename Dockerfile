FROM jupyter/r-notebook
USER jovyan
COPY . $HOME
RUN conda env create
SHELL ["conda", "init", "bash", "/bin/bash", "-c"]
RUN conda init bash
RUN echo "conda activate app" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

RUN conda activate app
RUN python run_simulation.py -h
RUN make -h
RUN cmake -h


