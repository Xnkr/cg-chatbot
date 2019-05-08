FROM python:3.6-slim

MAINTAINER Shankar "shankar.kanra@gmail.com"

# config
RUN mkdir -p /var/www 

# Define working directory.

COPY requirements.txt /var/www/requirements.txt

WORKDIR /var/www
RUN \
    pip3 install -r requirements.txt \
    && python3 -m spacy download en 

COPY . /var/www
RUN chmod -x index.py
EXPOSE 8009
ENTRYPOINT [ "python3" ]

CMD [ "index.py" ]
