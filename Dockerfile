FROM jupyter/r-notebook
USER jovyan
COPY . $HOME
WORKDIR $HOME
# ADD . ~/jovyan/work
# RUN echo $pwd
#RUN ls -ltr environment.yml
RUN conda env create && echo "conda activate app" >> ~/.profile
#RUN Rscript requirements.R
RUN $(which python3) run_simulation.py -h
RUN make -h
RUN cmake -h

RUN echo $(which python) 
RUN /opt/conda/bin/python3.10
#simulations/stdp.py
