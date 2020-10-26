FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

COPY utils.py ./utils.py
COPY preprocess_sequential.py ./preprocess_sequential.py

ENTRYPOINT ["conda", "run", "-n", "spotlight", "python", "preprocess_sequential.py"]
