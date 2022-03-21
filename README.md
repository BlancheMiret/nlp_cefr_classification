# NLP CEFR Classification

## Installation

- Create a virtual environment and activate it:

```
$ virtualenv -p python3 .env
$ source .env/bin/activate
```

- Install the requirements:

```
$ pip3 install -r requirements.txt
```

## Usage

```bash
$ cd src
$ ./run_all_experiments.sh
```

## Build Docker Image

```bash
$ export DOCKER_BUILDKIT=0    
$ docker build --no-cache -t docker_image .
```

## Run Docker Image

```bash
$ docker run -ti --rm --name='docker_image' docker_test:latest
```

