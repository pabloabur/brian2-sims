
# brian2-sims
Upstream: 
[![Docker Image CI](https://github.com/pabloabur/brian2-sims/actions/workflows/docker-image.yml/badge.svg)](https://github.com/pabloabur/brian2-sims/actions/workflows/docker-image.yml)

Here:
[![Docker Image CI](https://github.com/fun-zoological-computing/brian2-sims/actions/workflows/docker-image.yml/badge.svg)](https://github.com/fun-zoological-computing/brian2-sims/actions/workflows/docker-image.yml)

Install via `pip install -e .`. Also use requirements.txt to get dependencies. TODO: pip install -r requirements.txt instead? Seems to work

TODO: this needs double chekcing
If you are working on a remote server, conda might be a better option when linking other libraries/resources (e.g. C++). In that case, you can install miniconda and create my_env.yaml with the content

```
name: my_env
channels:
        - defaults
dependencies:
        - pip
        - pip:
                - -r requirements.txt
```

Install conda-build with `conda install conda-build`, and run `conda develop .`
