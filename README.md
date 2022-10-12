# brian2-sims
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
