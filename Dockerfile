# Dockerfile
FROM jupyter/tensorflow-notebook

USER root

RUN pip install --upgrade pip && \
    pip install transformers pysrt

RUN fix-permissions "/home/${NB_USER}"

USER $NB_UID
