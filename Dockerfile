FROM mambaorg/micromamba:1.3.1

USER root
COPY . .
RUN apt update && apt install -y cmake g++
RUN micromamba install -y -n base -f env.yaml \
    && micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -c 'import uuid; print(uuid.uuid4())' > /tmp/my_uuid

USER $MAMBA_USER
# TODO this is not really needed for build; it is a test. Go somewhere else
#RUN python run_simulation.py --quiet --backend cpp_standalone models
#RUN apt install -y r-base
