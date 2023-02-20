FROM jupyter/r-notebook
USER jovyan
COPY . $HOME
WORKDIR $HOME
# ADD . ~/jovyan/work
# RUN echo $pwd
RUN ls -ltr environment.yml
RUN conda env create && echo "conda activate app" >> ~/.profile
RUN Rscript requirements.R
ENTRYPOINT /bin/python ~/jovyan/work/brian2-sims/simulations/stdp.py
