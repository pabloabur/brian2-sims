[![Docker Image CI](https://github.com/pabloabur/brian2-sims/actions/workflows/docker-image.yml/badge.svg)](https://github.com/pabloabur/brian2-sims/actions/workflows/docker-image.yml)

# brian2-sims
First, build a container with 

```
docker build -t app .
```

then run the command below inside the project folder:

```
docker run -it --rm -v .:/home/app:rw app
```

The available simulations and their options will be printed, so you can run a simulation with e.g.

```
docker run -it --rm -v .:/home/app:rw app models
```

You might have to change permission for the volume that will be mounted (e.g. `chmod -R +2 brian2-sims`), as some simulations generate and save files.

The docker images were tagged and pushed to dockerhub, so you can also pull the image with `docker pull pabloabur/app` and run this image as shown above. Note that --gpu and --entrypoint flags can be used for using GPUs and probing the container, respectively. However, running with GPU support might require nvidia-container-toolkit.

We used conda-based environment because if you are working on a remote server, conda might be a better option when linking other libraries/resources (e.g. C++).

R was only used to generate fancy plots. I couldn't add R requirements to the dockerfile, but, if you wish to reproduce the plots, you can install the dependencies with `Rscript requirements.R` and run the plotting scripts outside the docker environment.
