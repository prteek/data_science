FROM python:3.11

COPY ./run_dbt.sh /usr/src/app/run_dbt.sh
WORKDIR /usr/src/app/

ARG GITHUB_TOKEN
ARG GITHUB_USER
ENV GITHUB_TOKEN=$GITHUB_TOKEN
ENV GITHUB_USER=$GITHUB_USER

RUN python -m pip install -U pip
RUN python -m pip install dbt dbt-athena-community

ENTRYPOINT ["/bin/bash", "./run_dbt.sh"]
