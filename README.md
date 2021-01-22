## FastAPI Machine learning apps 

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)                 
[![Python 3.6](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)

Just a collection of simple machine learning apps built and deployed using FastAPI and Python.

## Running the app 

**1.** Running the app locally on a development server

Clone the repository :- 
```
$ git clone https://github.com/rexsimiloluwah/fastapi-ml-apps.git
$ mkdir fastapi-ml-apps
```
Install requirements/ app dependencies :- 
```
$ pip install -r requirements.txt
```

Run the app :- 
```
$ cd app && python main.py
```
> View the app on http://localhost:5000

**2.** Running the app via Docker 
```
$ docker run --rm -p 5050:5050 similoluwaokunowo/fastapi-ml-apps
```

> View the app on http://localhost:5050

Pulling the docker image
```
$ docker pull similoluwaokunowo/fastapi-ml-apps
```

*Computer vision app is not live yet due to slug size issues with tensorflow and keras 

To view the live app :- 
> Go to https://fastapi-ml-apps.herokuapp.com

