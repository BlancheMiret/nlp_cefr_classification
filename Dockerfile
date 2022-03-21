#######################
# Choose an OS or runtime environment image that fits the needs of your experiment e.g.
#FROM debian:jessie
#Or:
FROM python:3.7.2
#FROM python
#######################

COPY . /app
RUN ls -la /app
RUN ls -la /usr/local/lib/
RUN pip3 install -r /app/requirements.txt
WORKDIR "/app/src/"
ENTRYPOINT ./run_docker.sh