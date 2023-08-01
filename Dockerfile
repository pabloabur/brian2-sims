FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

COPY . /app/
RUN apt update && apt install -y cmake g++ curl && apt clean \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj bin/micromamba \
    && micromamba install -y -n base -f /app/env.yaml -r /opt/micromamba \
    && micromamba clean --all --yes \
    && chmod -R o+rX /opt/micromamba \
    && micromamba shell init --shell bash -r /opt/micromamba

WORKDIR /app
ENTRYPOINT ["micromamba", "run", "-n", "base", "-r", "/opt/micromamba", "python", "run_simulation.py"]
CMD ["-h"]
