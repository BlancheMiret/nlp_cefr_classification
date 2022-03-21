# NLP CEFR Classification

This project had for goal to reproduce experiments published in https://aclanthology.org/W18-0515/.
Project was made by:
- Theo BOUGANIM
- Davide COLACI
- Blanche MIRET

## Usage with docker

### 1. Get docker image

**Option 1**: download the image and load it

- Download the image docker [here](https://www.swisstransfer.com/d/f2a85654-ece2-46b9-891f-92a9f38c9b67) and place it at the root of this project
- Execute:
```bash
$ docker load < docker_image.tar
```

**Option 2**: build the image yourself

```bash
$ cd '<chemin source projet>'
$ export DOCKER_BUILDKIT=0    
$ docker build --no-cache -t docker_image .
```

### 2. Run Docker Image

```bash
$ docker run -ti --rm --name='docker_image' docker_image:latest
```

And follow the instructions appearing on screen.

### (Export Docker Image)

```bash
$ docker save docker_image:latest > docker_image.tar
```

## Usage without docker

### 1. Installation

- Create a virtual environment and activate it:

```
$ virtualenv -p python3 .env
$ source .env/bin/activate
```

- Install the requirements:

```
$ pip3 install -r requirements.txt
```

### 2. Usage

```bash
$ cd src
$ ./run_all_experiments.sh
```
