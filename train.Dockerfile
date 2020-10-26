FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

COPY utils.py ./utils.py
COPY train.py ./train.py

ENTRYPOINT ["conda", "run", "-n", "spotlight", "python", "train.py"]
