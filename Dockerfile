FROM python:3.10-buster

RUN pip install pandas
RUN pip install catboost
RUN pip install pickle4

RUN mkdir -p /usr/src/app
COPY . /usr/src/app
WORKDIR /usr/src/app

ENTRYPOINT ["python", "Inference.py"]

