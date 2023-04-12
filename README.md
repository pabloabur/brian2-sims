[![Docker Image CI](https://github.com/pabloabur/brian2-sims/actions/workflows/docker-image.yml/badge.svg)](https://github.com/pabloabur/brian2-sims/actions/workflows/docker-image.yml)

# brian2-sims
First build container with 

```
docker build -t app .
```

then run it with 
```
docker run -it --rm -v $(pwd)/simulations:/tmp/simulations:ro -v $(pwd)/sim_data:/tmp/sim_data app
```

Type `python run_simulation -h` to see simulations available and their options.

We used conda-based environment because if you are working on a remote server, conda might be a better option when linking other libraries/resources (e.g. C++).

R was only used to generate fancy plots. I couldn't add R requirements to the dockerfile, but, if you wish to reproduce the plots, you can install the dependencies with `Rscript requirements.R` and run the plotting scripts outside the docker environment.
