# Using Python 3.7 Base docker image
FROM python:3.7-slim 

# Change working directory 
WORKDIR /usr/src/app 

# Copy the content of the app folder into the working directory
COPY ./app .

# Run system update 
RUN apt-get -y update

# Install dependencies 
RUN pip3 install -r requirements.txt

# Install dependencies of nltk stopwords and words corpora
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader words

# Run app
CMD ["python3", "main.py"]