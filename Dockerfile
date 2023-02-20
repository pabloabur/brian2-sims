FROM jupyter/r-notebook
COPY . ~
# ADD . ~/jovyan/work

RUN conda env create && echo "conda activate app" >> ~/.profile
RUN Rscript requirements.R
ENTRYPOINT /bin/python ~/jovyan/work/brian2-sims/simulations/stdp.py
