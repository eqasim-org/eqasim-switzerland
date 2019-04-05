FROM ubuntu:16.04

VOLUME ["/data"]

RUN apt-get update -y
RUN apt-get install curl bzip2 bash -y

COPY setup setup
RUN bash setup/setup.sh

COPY choice_model /root/choice_model
COPY data /root/data
COPY matsim /root/matsim
COPY population /root/population
COPY utils /root/utils
COPY pipeline.py /root/pipeline.py
COPY run.py /root/run.py

COPY docker_run.sh docker_run.sh
ENTRYPOINT ["bash", "docker_run.sh"]

VOLUME ["/cache"]
