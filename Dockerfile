FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

USER root

COPY env.yaml /tmp/env.yaml
RUN apt update && apt install -y cmake g++ curl \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj bin/micromamba

ARG user=appuser
ARG group=appuser
ARG uid=1001
ARG gid=1001
RUN groupadd -g ${gid} ${group}
RUN useradd -u ${uid} -g ${group} -s /bin/sh -m ${user} # <--- the '-m' create a user home directory

# Switch to user
USER ${uid}:${gid}

RUN micromamba install -y -n base -f /tmp/env.yaml \
    && micromamba clean --all --yes
RUN micromamba shell init --shell bash --root-prefix=~/micromamba
WORKDIR /home/app
ENTRYPOINT ["micromamba", "run", "-n", "base", "python", "run_simulation.py"]
CMD ["-h"]
