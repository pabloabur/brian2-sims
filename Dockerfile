FROM jupyter/r-notebook

COPY . ~/jovyan/work
WORKDIR ~/jovyan/work
RUN conda env create && echo "conda activate app" >> ~/.profile
RUN Rscript requirements.R
ENTRYPOINT /bin/python /brian2-sims/simulations/stdp.py
